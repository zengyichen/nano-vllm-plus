import time
import random
import tracemalloc
from collections import deque
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

def create_sequence(token_ids):
    # 创建序列
    seq = Sequence(token_ids, SamplingParams())
    return seq

def measure_time(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def benchmark_prefix_caching(num_blocks=10000, block_size=16):
    print("\n" + "="*50)
    print(f"🚀 运行 Benchmark: Prefix Caching (Shared Prompt)")
    print("="*50)
    
    Sequence.block_size = block_size
    manager = BlockManager(num_blocks, block_size)
    
    # 模拟一个巨大的 System Prompt
    shared_prompt_len = 4096
    shared_prompt = [random.randint(0, 32000) for _ in range(shared_prompt_len)]
    
    # 100 个并发请求，共享 system prompt，但 user query 不同
    num_seqs = 100
    user_query_len = 128
    
    print(f"📝 准备 {num_seqs} 个请求，每个请求包含 {shared_prompt_len} 共享 Token 和 {user_query_len} 独立 Token...")
    seqs = []
    for i in range(num_seqs):
        token_ids = shared_prompt + [random.randint(0, 32000) for _ in range(user_query_len)]
        seqs.append(create_sequence(token_ids))
        
    # 分配测试
    start = time.perf_counter()
    for seq in seqs:
        if manager.can_allocate(seq):
            manager.allocate(seq)
        else:
            print("❌ 空间不足，分配失败！(需要扩容或支持 Swapping)")
            break
    end = time.perf_counter()
    
    used_blocks = manager.block_size - len(manager.free_block_ids) if hasattr(manager, 'free_block_ids') else len(manager.used_block_ids)
    print(f"⏱️  分配耗时: {end - start:.4f} 秒")
    print(f"📊 已用 Block 数: {len(manager.used_block_ids)}")
    print(f"♻️  如果没有 Prefix Caching，理论应该使用 {(shared_prompt_len + user_query_len) // block_size * num_seqs} 个 Block")
    
    # 验证清理
    for seq in seqs:
        manager.deallocate(seq)
    print(f"🧹 清理后已用 Block: {len(manager.used_block_ids)} (应为 0)")

def benchmark_cow_and_append(num_blocks=10000, block_size=16):
    print("\n" + "="*50)
    print(f"🚀 运行 Benchmark: Copy-on-Write (CoW) & Append")
    print("="*50)
    
    Sequence.block_size = block_size
    manager = BlockManager(num_blocks, block_size)
    
    # 一个初始序列
    prompt_len = 512
    prompt = [random.randint(0, 32000) for _ in range(prompt_len)]
    seq = create_sequence(prompt)
    manager.allocate(seq)
    
    # 模拟 Beam Search 或 并行采样分裂出多个子序列
    branch_count = 100
    decode_steps = 128
    
    print(f"🌿 模拟从 {prompt_len} 个 Token 的 Prompt 并行生长出 {branch_count} 个分支，每个分支生成 {decode_steps} 步...")
    
    # 给未来的 CoW 优化做性能对比：目前没有完整的共享树逻辑，只能拷贝 Token 建新序列
    branches = []
    start_clone = time.perf_counter()
    for i in range(branch_count):
        # 假设每次都深拷贝一份 prompt（目前实现的痛点）
        b_seq = create_sequence(prompt.copy())
        manager.allocate(b_seq)
        branches.append(b_seq)
    end_clone = time.perf_counter()
    
    print(f"⏱️  分支克隆和初始 Allocate 耗时: {end_clone - start_clone:.4f} 秒")
    
    # 模拟自回归生成
    start_decode = time.perf_counter()
    for step in range(decode_steps):
        for b_seq in branches:
            # 增加一个新 Token
            b_seq.append_token(random.randint(0, 32000))
            if manager.can_append(b_seq):
                manager.may_append(b_seq)
            else:
                pass # 忽略报错，假装 Swap 发生了
    end_decode = time.perf_counter()
    t = end_decode - start_decode
    total_tokens = decode_steps * branch_count
    throughput = total_tokens / t
    
    print(f"⏱️  自回归解码 ({decode_steps} 步 x {branch_count} 分支) 耗时: {t:.4f} 秒")
    print(f"📊 最终使用的 Block 数量: {len(manager.used_block_ids)}")
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


from nanovllm.engine.scheduler import Scheduler

class MockConfig:
    def __init__(self, num_blocks, block_size):
        self.max_num_seqs = 2000
        self.max_num_batched_tokens = 999999
        self.eos = -1
        self.num_kvcache_blocks = num_blocks
        self.kvcache_block_size = block_size

def benchmark_memory_thrashing_swapping(num_blocks=5000, block_size=16):
    print("\n" + "="*50)
    print(f"🚀 运行 Benchmark: Memory Thrashing (利用 Scheduler 触发真实抢占)")
    print("="*50)
    
    Sequence.block_size = block_size
    config = MockConfig(num_blocks, block_size)
    scheduler = Scheduler(config)
    
    req_len = 1000
    decode_steps = 200
    num_requests_total = 200
    
    print(f"🌊 将 {num_requests_total} 个请求塞入 Scheduler 并持续调度。")
    print(f"每个请求 Prompt 长度 {req_len}，将持续 Decode 生成 {decode_steps} 步。")
    print(f"显存池限制为 {num_blocks} 块，容量严重不足，必然引发反复的 Preempt(抢占)！...")

    # 劫持一下 preempt 函数用来精确计数真实的颠簸次数
    preempt_count = 0
    original_preempt = scheduler.preempt

    def mock_preempt(seq):
        nonlocal preempt_count
        preempt_count += 1
        original_preempt(seq)
        
    scheduler.preempt = mock_preempt

    for i in range(num_requests_total):
        params = SamplingParams()
        params.max_tokens = decode_steps
        seq = Sequence([random.randint(0, 32000) for _ in range(req_len)], params)
        scheduler.add(seq)
        
    start_time = time.perf_counter()
    
    step = 0
    total_generated_tokens = 0
    while not scheduler.is_finished():
        scheduled_seqs, is_prefill = scheduler.schedule()
        if not scheduled_seqs:
            # 空间过小，甚至连一个 prefill 都满足不了，直接跳出避免死循环
            break
            
        # 模拟模型执行完毕并返回新的 token
        # 对每一个正在跑的序列追加一个 token 触发下一步的空间压缩
        token_ids = [random.randint(0, 32000) for _ in scheduled_seqs]
        scheduler.postprocess(scheduled_seqs, token_ids)
        total_generated_tokens += len(scheduled_seqs)
        step += 1
            
    end_time = time.perf_counter()
    t = end_time - start_time
    throughput = total_generated_tokens / t if t > 0 else 0
    
    print(f"⏱️  总耗时 ({step} 轮调度): {t:.4f} 秒")
    print(f"⚠️  由 Scheduler 真实触发的抢占下放 (Eviction/Deallocation) 次数: {preempt_count}")
    print(f"Total: {total_generated_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    print("💡 目标提示：当前抢占会清空 Sequence 所有的 Block！")
    print("加入 Swapping 机制或 Paged Cache 管理可以避免重新计算。")


if __name__ == "__main__":
    benchmark_prefix_caching()
    benchmark_cow_and_append()
    benchmark_memory_thrashing_swapping()
    print("\n✅ Benchmark 完成。")

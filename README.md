# AI_infra_learn
AI Infra Learning Journal

## CUDA add 实验总结--1 day
在这次实现的两个版本中，naive 版本的耗时为 0.095 ms，shared-memory 版本为 0.204 ms。结果表明，在当前这个简单的 1 block / 1024 threads 向量加法场景下，shared memory 并没有带来性能提升，反而因为额外的同步和数据搬运开销使运行时间更长。这也说明：不是所有场景都适合使用 shared memory，真正的性能收益取决于问题规模和访存模式。



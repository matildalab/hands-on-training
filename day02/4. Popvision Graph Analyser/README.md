# Popvision Graph Analyser

PopVision™ is a suite of graphical application-analysis tools. The PopVision™ Graph Analyser helps you get a deep understanding of how your applications are performing and utilising the IPU resources. It shows data about the graph program, memory use, and the time spent on code execution and communication.

## Download

You can download the tool from our [Tools page](https://www.graphcore.ai/developer/popvision-tools). Choose appropriate OS and download Graph Analyser.

## Generate a Report

To generate a profile report to be analysed with the tool, you need to run with an environmet variable `POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./reports", "debug.allowOutOfMemory":"true"}'`.
```bash
(venv_pytorch)$ POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./reports", "debug.allowOutOfMemory":"true"}' python model_parallel_pytorch.py &> model_parallel.log &
```
After running this command, the report will be generated in `reports` directory.
```
reports
├── debug.cbor
└── training
    ├── archive.a
    ├── debug.cbor -> /localdata/home/trainee01/hands-on-training/day02/4. Popvision Graph Analyser/./reports/debug.cbor
    └── profile.pop
```
**Note that you should generate this report only when you want to see the profile, because while generating the report, overall performance of the application gets lower. So, if you run an application where performance is critical, you must run without this setting.**

### Summary
Summary of the IPU hardware, graph parameters, host configuration.

![Summary](./images/summary.png?raw=true "Summary")

### Insights
Insights report gives you a quick overview of the memory usage of your model on the IPU, showing the tiles, vertices and exchanges that use the most memory.

![Insights](./images/insights.png?raw=true "Insights")

### Memory Report
Memory report gives a detailed analysis of memory usage across all the tiles in your IPU system, showing graphs of total memory and liveness data, and details of variable types, placement and size.

![Memory](./images/memory.png?raw=true "Memory")

### Liveness Report
Liveness report gives a detailed breakdown of the state of the variables at each step in your program.

![Liveness](./images/liveness.png?raw=true "Liveness")

### Execution Trace
Execution trace shows how many cycles each step of your instrumented program consumes.

![Execution](./images/execution.png?raw=true "Execution")

For more details, please visit our [Popvision User Guide](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/introduction.html).

[Continue to the next chapter](../../day03)
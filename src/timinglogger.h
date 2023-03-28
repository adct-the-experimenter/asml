///////////////////////////////////////////
//
// timinglogger - accumulate accurate execution times and log them to file
//
// Author: Daniel Paulsen
// University of Arizona
// Dept. of Electrical and Computer Engineering
// ECE 569, Group 2, Term Project
//
// 20230310 - Genesis
//
// Usage notes: Assumes serial execution
//
///////////////////////////////////////////
#ifndef TIMINGLOGGER_H
#define TIMINGLOGGER_H

// toggle logging
#define DEBUG_LOG_FLAG 1
// turn on the detailed log file output
#define LOG_DETAIL_FLAG 0
// turn on the rollup log file output
// this summarizes by the log message (# of executions, total, average, min, and max exectution time)
#define LOG_ROLLUP_FLAG 1
// LOG_TIME_UNIT can be nanoseconds, microseconds, seconds, etc
#define LOG_TIME_UNIT nanoseconds
#define LOG_TIME_UNIT_STR "nanoseconds"

#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <functional>
#include <boost/bind/bind.hpp>

using namespace std;
using namespace std::chrono;

class ExecTimeLogger {
public:
	void logStart(string processName);
	void logStop(string processName);
	void printLog();
	virtual ~ExecTimeLogger();
	std::string GetCurrentTimeForFileName();
private:
	//log detail
	struct ExecTimeScalar {
		string processName;
		std::chrono::high_resolution_clock::time_point startTime;
		std::chrono::high_resolution_clock::time_point endTime;
		std::chrono::duration<double> execDuration;
	};	
	vector<ExecTimeScalar> ExecTimeLog;
	//log rollup
	struct ExecTimeRollupScalar {
		string processName;
		int numExecutions = 0;
		std::chrono::duration<double> totalDuration;
		std::chrono::duration<double> avgDuration;
		std::chrono::duration<double> maxDuration;
		std::chrono::duration<double> minDuration;
	};	
	vector<ExecTimeRollupScalar> ExecTimeLogRollup;
};

extern ExecTimeLogger logExecTimes;

#endif //TIMINGLOGGER_H
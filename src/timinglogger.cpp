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
// Usage notes: 1. messages with the same label cannot overlap (one must complete before an 
//				   identical message is logged or timings are not recorded correctly) except for recursive calls
//
// syntax: logStart("event being logged"); //records starting timestamp
//         logEnd("event being logged"); //appends ending timestamp and duration
//         printLog(); //prints all recorded events and durations to a file.
//
///////////////////////////////////////////

#include "timinglogger.h"

void ExecTimeLogger::logStart(string pName) {
	if (DEBUG_LOG_FLAG) {
		ExecTimeScalar logEntry;
		logEntry.processName = pName;
		logEntry.startTime = std::chrono::high_resolution_clock::now();
		ExecTimeLogger::ExecTimeLog.push_back(logEntry);
	}
	return;
}
void ExecTimeLogger::logStop(string pName){
	if (DEBUG_LOG_FLAG) {
		auto endTimeStamp = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::time_point epochTime; //set to epoch time by default
		//start looking for "open" log messages at the end of the log array
		for (auto i = ExecTimeLogger::ExecTimeLog.rbegin(); i != ExecTimeLogger::ExecTimeLog.rend(); ++i ) {
			//append data to the last "open" (i.e., no duration) message with the same label.
			if (i->processName == pName && i->endTime == epochTime) {
				i->endTime = endTimeStamp;
				i->execDuration = endTimeStamp - i->startTime; //calculate the duration
				break;
			}
		}
		return;
	}
}
void ExecTimeLogger::printLog(){
	if (DEBUG_LOG_FLAG) {
		// print the minimum representable duration
		//std::cout << (double) std::chrono::high_resolution_clock::period::num / std::chrono::high_resolution_clock::period::den << endl;
		auto fileNameTimeStamp = GetCurrentTimeForFileName();
		string logTimeUnits = LOG_TIME_UNIT_STR;
//		std::stringstream ss;
//		ss << LOG_TIME_UNIT_STR;
//		ss >> logTimeUnits;
		if (LOG_DETAIL_FLAG) {
			//declare variables
			ofstream logFile;
			string logFileName = "ASML_EXEC_LOG_" + fileNameTimeStamp + ".txt";
			logFile.open(logFileName);
			if (logFile.is_open()) {
				string logLine = "ProcessName\tDuration_"+logTimeUnits;
				string exeDurationStr;
				logFile << logLine << endl;
				for (auto vectorit = ExecTimeLog.begin(); vectorit != ExecTimeLog.end(); ++vectorit) {
					auto exeDuration = std::chrono::duration_cast< std::chrono::LOG_TIME_UNIT >( (*vectorit).execDuration );
					exeDurationStr = to_string(exeDuration.count());
					logLine = (*vectorit).processName + "\t" + exeDurationStr;
					logFile << logLine << endl;
				}
				logFile.close();
			}
		}
		if (LOG_ROLLUP_FLAG) {
			//declare variables
			for (auto vectorit = ExecTimeLog.begin(); vectorit != ExecTimeLog.end(); ++vectorit) {
				if (ExecTimeLogRollup.empty()) {
					ExecTimeRollupScalar newRow;
					newRow.processName = vectorit->processName;
					newRow.numExecutions = 1;
					newRow.totalDuration = vectorit->execDuration;
					newRow.maxDuration = vectorit->execDuration;
					newRow.minDuration = vectorit->execDuration;
					ExecTimeLogRollup.push_back(newRow);
				} else {
					auto rollupRow = std::find_if ( ExecTimeLogRollup.begin (), ExecTimeLogRollup.end (), 
						[&](const auto& e) { return e.processName == (*vectorit).processName; });
					if (rollupRow != end(ExecTimeLogRollup)) {
						rollupRow->numExecutions += 1;
						rollupRow->totalDuration += vectorit->execDuration;
						if (vectorit->execDuration > rollupRow->maxDuration) {
							rollupRow->maxDuration = vectorit->execDuration;
						}
						if (vectorit->execDuration < rollupRow->minDuration) {
							rollupRow->minDuration = vectorit->execDuration;
						}
					} else {
						ExecTimeRollupScalar newRow;
						newRow.processName = vectorit->processName;
						newRow.numExecutions = 1;
						newRow.totalDuration = vectorit->execDuration;
						newRow.maxDuration = vectorit->execDuration;
						newRow.minDuration = vectorit->execDuration;
						ExecTimeLogRollup.push_back(newRow);
					}
				}
			}
			for (auto vectorit = ExecTimeLogRollup.begin(); vectorit != ExecTimeLogRollup.end(); ++vectorit) {
				vectorit->avgDuration = vectorit->totalDuration / vectorit->numExecutions;
			}
			ofstream logRollupFile;
			string logRollupFileName = "ASML_EXEC_LOG_ROLLUP_" + fileNameTimeStamp + ".txt";
			logRollupFile.open(logRollupFileName);
			if (logRollupFile.is_open()) {
				string logRollupLine = "ProcessName\tTotalExecutions\tTotalDuration_"+logTimeUnits+"\tAvgDuration_"+logTimeUnits+"\tMinDuration_"+logTimeUnits+"\tMaxDuration_"+logTimeUnits;
				logRollupFile << logRollupLine << endl;
				string numExecutionsStr;
				string totDurationStr;
				string avgDurationStr;
				string minDurationStr;
				string maxDurationStr;
				for (auto vectorit = ExecTimeLogRollup.begin(); vectorit != ExecTimeLogRollup.end(); ++vectorit) {
					auto totDur = std::chrono::duration_cast< std::chrono::LOG_TIME_UNIT >( (*vectorit).totalDuration );
					auto avgDur = std::chrono::duration_cast< std::chrono::LOG_TIME_UNIT >( (*vectorit).avgDuration );
					auto minDur = std::chrono::duration_cast< std::chrono::LOG_TIME_UNIT >( (*vectorit).minDuration );
					auto maxDur = std::chrono::duration_cast< std::chrono::LOG_TIME_UNIT >( (*vectorit).maxDuration );
					numExecutionsStr = to_string((*vectorit).numExecutions);
					totDurationStr = to_string(totDur.count());
					avgDurationStr = to_string(avgDur.count());
					minDurationStr = to_string(minDur.count());
					maxDurationStr = to_string(maxDur.count());
					logRollupLine = (*vectorit).processName + "\t" + numExecutionsStr + "\t" + totDurationStr + "\t" + avgDurationStr + "\t" + minDurationStr + "\t" + maxDurationStr;
					logRollupFile << logRollupLine << endl;
				}
				logRollupFile.close();
			}
		}
	}
	return;
}

ExecTimeLogger::~ExecTimeLogger(){}; //destructor

// below function was obtained from https://stackoverflow.com/questions/22318389/pass-system-date-and-time-as-a-filename-in-c
std::string ExecTimeLogger::GetCurrentTimeForFileName() {
    auto time = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%F_%T"); // ISO 8601 without timezone information.
    auto s = ss.str();
    std::replace(s.begin(), s.end(), ':', '-');
    return s;
}

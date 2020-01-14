#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <thread>
typedef void(*callback_routines)();

std::vector<void(*)(std::vector<std::string>*)> command_callbacks;
std::thread command_parser_daemon;
std::thread::id main_thread_id;

std::mutex message_mutex;
std::vector<int> message_queue;

class CallbackFuncs {
public:
	const int id;
	const std::vector<std::string>* params;
	callback_routines *sync_routine, *async_routine;
	
	CallbackFuncs(const int id, std::vector<std::string>* params, void(*)()) :id(id), params(params) {

	}
};

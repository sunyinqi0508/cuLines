#ifndef _H_COMMAND_PARSER
#define _H_COMMAND_PARSER
#include <vector>
#include <string>
#include <mutex>
#include <thread>


typedef void(*callback_routine)();

std::vector<void(*)(std::vector<std::string>*)> command_callbacks;
std::thread command_parser_daemon;
std::thread::id main_thread_id;

std::mutex message_mutex;
std::vector<int> message_queue;

class CallbackFuncs {
public:
	const int id;
	const std::vector<std::string>* params;
	callback_routine sync_routine, async_routine;
	
	CallbackFuncs(const int id, std::vector<std::string>* params, callback_routine sync_routine, callback_routine async_routine) :id(id), params(params) {
		if (sync_routine)
			this->sync_routine = sync_routine;
		else
			;
	}
};
#endif

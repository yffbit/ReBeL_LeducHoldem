#if !defined(_THREAD_LOOP_H_)
#define _THREAD_LOOP_H_

#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
using std::vector;
using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::unique_lock;
using std::lock_guard;
using std::thread;
using std::shared_ptr;
using std::make_shared;

class ThreadLoop {
public:
    ThreadLoop() = default;
    virtual void loop() {
        // while(run) {
        //     wait_until_resume();
        //     if(!run) break;
        //     // code
        // }
    }
    void pause() {// 暂停loop
        lock_guard<mutex> lg(mt_paused);
        paused = true;
    }
    void resume() {// 继续loop
        {
            lock_guard<mutex> lg(mt_paused);
            paused = false;
        }
        cv_paused.notify_one();
    }
    void stop() {// 结束loop
        run = false;
        resume();// 先唤醒正在等待的线程
    }
protected:
    atomic_bool run = true;
    bool paused = false;
    mutex mt_paused;
    condition_variable cv_paused;
    void wait_until_resume() {
        unique_lock<mutex> lk(mt_paused);
        cv_paused.wait(lk, [this](){return !paused;});
    }
};

class Context {
public:
    Context():started(false) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() {
        stop();
        join();
    }
    void push(shared_ptr<ThreadLoop> loop) {
        if(started) throw std::runtime_error("you should push all work loop before calling start()");
        loops.push_back(move(loop));
    }
    void start() {
        int n = loops.size();
        for(int i = 0; i < n; i++) {
            threads.emplace_back([this, i]() {
                loops[i]->loop();
            });
        }
        started = true;
    }
    void stop() {
        int n = loops.size();
        for(int i = 0; i < n; i++) loops[i]->stop();
    }
    void join() {
        int n = threads.size();
        for(int i = 0; i < n; i++) threads[i].join();
        threads.clear();
        loops.clear();
        started = false;
    }
private:
    bool started;
    vector<shared_ptr<ThreadLoop>> loops;
    vector<thread> threads;
};

#endif // _THREAD_LOOP_H_

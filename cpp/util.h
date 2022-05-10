#if !defined(_UTIL_H_)
#define _UTIL_H_

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
using std::vector;
using std::string;
using namespace std::chrono;

class Timer {
    steady_clock::time_point t0;
public:
    Timer() {
        start();
    }
    void start() {
        t0 = steady_clock::now();
    }
    double duration() {
        steady_clock::time_point t1 = steady_clock::now();
        microseconds s = duration_cast<microseconds>(t1 - t0);
        return s.count();
    }
};

template<class T>
class Combination {
public:
    Combination(vector<T>& candidate, int k):candidate(candidate), temp(k), K(k) {
        N = candidate.size();
        if(N >= k && k > 0) backtrace(0, 0);
    }
    const vector<vector<T>>& get_result() {
        return result;
    }
private:
    void backtrace(int i, int j) {// 候选集合已遍历i个,已选择j个
        if(j == K) {
            result.push_back(temp);
            return;
        }
        temp[j] = candidate[i];// 选
        backtrace(i+1, j+1);
        if(N - i <= K - j) return;// 不能不选
        backtrace(i+1, j);// 不选
    }

    int N = 0, K = 0;
    vector<T>& candidate;
    vector<T> temp;
    vector<vector<T>> result;
};

// 组合数C(n,k)
long long comb(long long n, long long k) {
    if(k < 0 || k > n) return 0;
    if(k == 0 || k == n) return 1;
    long long ans = n;
    for(long long d = 2LL; d <= k; d++) ans = ans * (--n) / d;
    return ans;
}

// 最低位的1对应的mask
template<class T>
T low_bit(T x) {
    // return x & (-x);
    return x & (~(x - 1));
}

inline int popcnt(uint32_t x) {
    x -= (x >> 1) & 0x55555555;
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x += x >> 8;
    return (x + (x >> 16)) & 0x3F;
}

inline int popcnt(uint64_t x) {
    return popcnt((uint32_t)x) + popcnt((uint32_t)(x >> 32));
}

template<class T>
class MaskCombination {
public:
    MaskCombination(T mask, int n_bit, int k) {
        T MASK = ((T)1 << n_bit) - 1;
        mask &= MASK;
        int n = 0;
        if(sizeof(mask) <= sizeof(uint32_t)) n = popcnt((uint32_t)mask);
        else n = popcnt((uint64_t)mask);
        if(k <= 0 || k > n) return;
        backtrace(mask, n, 0, k);
    }
    const vector<T>& get_result() {
        return result;
    }
private:
    void backtrace(T mask, int n, T curr, int k) {
        if(k == 0) {
            result.push_back(curr);
            return;
        }
        for(T p = 0; n >= k; ) {
            p = low_bit<T>(mask);// 优先选择最低位的1
            mask ^= p;
            backtrace(mask, --n, curr|p, k-1);
        }
    }
    
    vector<T> result;
};
template MaskCombination<int>;
template MaskCombination<long long>;

template<class T>
void print(const vector<T>& vec, char delimiter=',') {
    std::cout << '[';
    int n = vec.size();
    if(n) {
        std::cout << vec[0];
        for(int i = 1; i < n; i++) std::cout << ',' << vec[i];
    }
    std::cout << ']' << std::endl;
}

template<class T>
void print(const vector<vector<T>>& vecs, char delimiter=',') {
    std::cout << '[' << std::endl;
    for(const vector<T>& vec : vecs) print(vec, delimiter);
    std::cout << ']' << std::endl;
}

// 字符串分割
template<class T>
vector<string> string_split(string& s, T delimiter, int offset = 0) {
    vector<string> ans;
    size_t i = s.find_first_not_of(delimiter, offset);
    while (i != string::npos) {
        size_t j = s.find_first_of(delimiter, i + 1);
        ans.emplace_back(move(s.substr(i, j - i)));
        i = s.find_first_not_of(delimiter, j);
    }
    return ans;
}
template vector<string> string_split<string>(string&, string, int);
template vector<string> string_split<char>(string&, char, int);

long long fast_power(int x, int n) {
    long long ans = 1;
    while(n) {
        if(n&1) ans *= x;
        x *= x;
        n >>= 1;
    }
    return ans;
}

#endif // _UTIL_H_

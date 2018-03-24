#include <limits>
#include <cassert>
#include <cmath>

template <typename T = float>
constexpr T EPSILON = static_cast<T>(1E-6);

template <typename T = float>
constexpr T PI = 3.14159265358979323846264338327950288;

template <typename T>
constexpr T cubic(T x) {
    return x * x * x;
}

struct alignas(float) Vector {
	float x = 0.f;
    float y = 0.f;
    float z = 0.f;
	
	Vector(float v) { x = y = z = v; }
	Vector() = default;
	Vector(float x, float y, float z) : x(x), y(y), z(z) {  }
	inline operator float() { return x + y + z; }
    inline float operator[] (int index) const {
        assert(0 <= index && index < 3);
        return reinterpret_cast<const float*>(this)[index];
    }

    inline float &operator[] (int index) {
        assert(0 <= index && index < 3);
        return reinterpret_cast<float*>(this)[index];
    }

    inline Vector &operator+= (const Vector &rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
    inline Vector &operator-= (const Vector &rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    inline Vector &operator*= (const Vector &rhs) {
        x *= rhs.x;
        y *= rhs.y;
        z *= rhs.z;
        return *this;
    }

    inline Vector &operator/= (const Vector &rhs) {
        x /= rhs.x;
        y /= rhs.y;
        z /= rhs.z;
        return *this;
    }

    inline Vector &operator+= (float value) {
        x += value;
        y += value;
        z += value;
        return *this;
    }

    inline Vector &operator-= (float value) {
        x -= value;
        y -= value;
        z -= value;
        return *this;
    }

    inline Vector &operator*= (float value) {
        x *= value;
        y *= value;
        z *= value;
        return *this;
    }

    inline Vector &operator/= (float value) {
        x /= value;
        y /= value;
        z /= value;
        return *this;
    }
    inline Vector operator+ (const Vector &rhs) const {
        return Vector{ x + rhs.x, y + rhs.y, z + rhs.z };
    }

    inline Vector operator- (const Vector &rhs) const {
        return Vector{ x - rhs.x, y - rhs.y, z - rhs.z };
    }

    inline Vector operator* (const Vector &rhs) const {
        return Vector{ x * rhs.x, y * rhs.y, z * rhs.z };
    }

    inline Vector operator/ (const Vector &rhs) const {
        return Vector{ x / rhs.x, y / rhs.y, z / rhs.z };
    }

    inline Vector operator+ (float value) const {
        return Vector{ x + value, y + value, z + value };
    }

    inline Vector operator- (float value) const {
        return Vector{ x - value, y - value, z - value };
    }

    inline Vector operator* (float value) const {
        return Vector{ x * value, y * value, z * value };
    }

    inline Vector operator/ (float value) const {
        return Vector{ x / value, y / value, z / value };
    }

    inline bool nearlyNull() const {
        return std::fabsf(x) < EPSILON<> &&
               std::fabsf(y) < EPSILON<> &&
               std::fabsf(z) < EPSILON<>;
    }

    inline Vector normalize() const {
        return *this / l2Norm();
    }

    
	inline float l1Norm() const {
		return fabsf(x) + fabsf(y) + fabsf(z);
	}
    inline float l2Norm() const {
        return std::sqrtf(x * x + y * y + z * z);
    }
	
	inline float sq() const {
		return x * x + y * y + z * z;
	}
    inline float dot(const Vector &rhs) const {
        return (*this * rhs);
    }

    inline Vector cross(const Vector &rhs) const {
        return {
            y * rhs.z - z * rhs.y,
            z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x
        };
    }

    inline float project(const Vector &target) const {
        return dot(target) / target.l2Norm();
    }

    inline float distance(const Vector &rhs) const {
        return (*this - rhs).l2Norm();	
    }

    inline float angle(const Vector &rhs) const {
        return std::acosf(dot(rhs) / (l2Norm() * rhs.l2Norm()));
    }
};

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

template <typename S, typename T>
bool lessFirst(const std::pair<S, T> &lhs, const std::pair<S, T> &rhs) {
    return lhs.first < rhs.first;
}

template <typename S, typename T>
bool greaterFirst(const std::pair<S, T> &lhs, const std::pair<S, T> &rhs) {
    return lhs.first > rhs.first;
}

struct Curve;
struct Segment;
struct Shape;

struct Point : public Vector {
    int index = -1;
    Curve *curve = nullptr;

    Point(Curve *crv, Vector v);
};

enum class DecompositionPolicy {
    Curvature, CurveSegmentRatio, Evenly
};

struct Curve {
private:
    void decomposeEvenly(float mean_len, float len_thresh);
    void decomposeByCurvature(float crv_thresh, float len_thresh);
	void decomposeGlobal(float, float penalty);
    void decomposeByRatio(float ratio_thresh, float len_thresh);
public:
    int index;
    std::vector<Point*> samples;
    std::vector<Segment*> segments;
    Shape *shape;

    Curve(Shape *shape, std::size_t idx) : shape{ shape }, index{ static_cast<int>(idx) } {}

    void decompose(DecompositionPolicy policy, float arg1, float arg2) {
		//decomposeGlobal(arg1, .95);
		//decomposeByRatio(arg1, arg2);
		decomposeByCurvature(arg1, arg2);
		//decomposeEvenly(arg1, arg2);

		return; 
        switch (policy) {
        case DecompositionPolicy::Curvature:
            decomposeByCurvature(arg1, arg2);
            break;
        case DecompositionPolicy::CurveSegmentRatio:
            decomposeByRatio(arg1, arg2);
            break;
        case DecompositionPolicy::Evenly:
            decomposeEvenly(arg1, arg2);
            break;
        }
    }
};

class SegmentPointLookupTable;

struct Segment {
    Curve *curve;
    SegmentPointLookupTable *table;
    Vector head, tail, centroid;
    int begin, end, size; // size == end - begin
    bool stop_by_length = false;

    Segment(Curve *crv, int b, int e)
        : curve{ crv }, begin{ b }, end{ e }, size{ e - b },
          head{ *crv->samples[b] }, tail{ *crv->samples[e - 1] } {
        for (int i = begin; i < end; i++)
            centroid += *curve->samples[i];
        centroid /= static_cast<float>(size);
		Vector center = centroid - (tail + head) / 2.f;
		tail += center;
		head += center;
    }

    Point *findNearestPoint(const Vector &s);
};

inline std::vector<Point*>::iterator begin(Segment *seg) {
    return seg->curve->samples.begin() + seg->begin;
}

inline std::vector<Point*>::iterator end(Segment *seg) {
    return seg->curve->samples.begin() + seg->end;
}

class SegmentPointLookupTable {
    Segment *seg_;
    Vector target_;
    float min_, max_, width_;
    std::vector<Point*> map_;
public:
    SegmentPointLookupTable(Segment *seg) : seg_{ seg }, target_{ seg->tail - seg->head } {
       // target_ = seg->tail - seg->head;
        std::vector<std::pair<float, Point*>> proj;
        proj.reserve(seg->size);
        for (auto pt : seg)
            proj.emplace_back((*pt - seg->head).project(target_), pt);
        std::sort(begin(proj), end(proj));
        map_.resize(proj.size() * 5);
        min_ = proj.front().first;
        max_ = proj.back().first;
        width_ = (max_ - min_) / static_cast<float>(map_.size());
        std::vector<std::size_t> mid;
        mid.reserve(proj.size());
        for (std::size_t i = 1; i < proj.size(); i++)
            mid.emplace_back(static_cast<std::size_t>(std::floor((proj[i].first + proj[i - 1].first) / 2 / width_)));
        mid.emplace_back(map_.size());
        std::size_t j = 0;
        for (std::size_t i = 0; i < map_.size(); i++) {
            if (i < mid[j])
                map_[i] = proj[j].second;
            else
                map_[i] = proj[++j].second;
        }
    }

    Point *nearest(const Vector &v) const {
		seg_->centroid;
        auto p = (v - seg_->head).project(target_);
        if (p - min_ < EPSILON<>)
            return map_.front();
        if (max_ - p < EPSILON<>)
            return map_.back();
        return map_[static_cast<std::size_t>(std::floor((p - min_) / width_))];
    }
};

// type QueryResult = [(float, Point)]
typedef std::vector<std::pair<float, Point*>> QueryResult;


// a `Shape` is used for representing a 3-D stream-line object.
struct Shape {
    // all curves i.e. stream lines
    std::vector<Curve*> curves;
    // sample points on all curves
    std::vector<Point*> points;
    // line segments on all curves
    std::vector<Segment*> segments;

    // dispose all resources it has and make it empty
    void tidy() {
        points.clear();
        for (auto crv : curves)
            delete crv;
        curves.clear();
        segments.clear();
    }

    ~Shape() {
        tidy();
    }

    void loadVortexFile(const char *filename) {
        tidy();
        FILE *fp;
        if (fopen_s(&fp, filename, "r"))
            return;
        fclose(fp);
    }

    // load data from a WaveFront object file (extenstion name `.obj`)
    void load(const char *filename,
              DecompositionPolicy decompose_policy,
              std::pair<float, float> decompose_args,
              bool do_translate = false) {
        Vector cell_lengths{ 0.5f, 0.503937f, 0.516129f };
        Vector origins = { -64.f, -32.f, -8.f };
        tidy();
        FILE *fp;
        if (fopen_s(&fp, filename, "r"))
            return;
        int c;
        curves.emplace_back(new Curve{ this, curves.size() });
        while ((c = fgetc(fp)) != EOF) {
            if (c == 'v' && fgetc(fp) != 't') {
                Vector v;
                fscanf_s(fp, "%f%f%f", &v.x, &v.y, &v.z);
                if (do_translate)
                    v = v * cell_lengths + origins;
                // make sure this is not a duplicated point
                if (curves.back()->samples.empty() || v.distance(*curves.back()->samples.back()) > EPSILON<>) {
                    auto pt = new Point{ curves.back(), v };
                    curves.back()->samples.emplace_back(pt);
                    points.emplace_back(pt);
                }
            } else if (c == 'l') {
                curves.back()->decompose(decompose_policy, decompose_args.first, decompose_args.second);
                curves.emplace_back(new Curve{ this, curves.size() });
            }
            // move file cursor to the beginning of next line or end of file
            while (c = fgetc(fp), c != '\n' && c != EOF);
        }
        // remove last empty curve
        if (curves.back()->samples.empty()) {
            delete curves.back();
            curves.pop_back();
        }
        fclose(fp);
    }

    QueryResult find(const Vector &s, std::size_t k) {
        // find the nearest segment on all curves
        QueryResult pts;
        std::transform(begin(curves), end(curves), std::back_inserter(pts), [s](Curve *crv) {
            auto seg = *std::min_element(begin(crv->segments), end(crv->segments), [s](Segment *lhs, Segment *rhs) {
                return s.distance(lhs->centroid) < s.distance(rhs->centroid);
            });
            auto pt = seg->findNearestPoint(s);
            return std::make_pair(pt->distance(seg->centroid), pt);
        });
        return pts;
    }
};

void Curve::decomposeEvenly(float mean_len, float unused) {
    auto len_sum = 0.f;
    for (std::size_t i = 1; i < samples.size(); i++)
        len_sum += samples[i]->distance(*samples[i - 1]);
    auto n = static_cast<std::size_t>(samples.size() / (len_sum / mean_len));
    for (std::size_t i = 0; i < samples.size(); i += n) {
        auto end = std::min(i + n + 1, samples.size());
        auto seg = new Segment{ this, static_cast<int>(i), static_cast<int>(end) };
        segments.emplace_back(seg);
        shape->segments.emplace_back(seg);
        seg->table = new SegmentPointLookupTable{ seg };
    }
}

void Curve::decomposeByCurvature(float crv_thresh, float len_thresh) {
    // the first derivative
    std::vector<Vector> r;
    r.reserve(samples.size() - 1);
    for (std::size_t i = 1; i < samples.size(); i++)
        r.emplace_back(*samples[i] - *samples[i - 1]);
    // the second derivative
    std::vector<Vector> rr;
    rr.reserve(samples.size() - 2);
    for (std::size_t i = 1; i < r.size(); i++)
        rr.emplace_back(r[i] - r[i - 1]);
    // the curvature
    std::vector<float> kappa;
    kappa.reserve(samples.size() - 2);
	for (std::size_t i = 0; i < rr.size(); i++)
		kappa.emplace_back(r[i].cross(rr[i]*(r[i])).l2Norm() / cubic(r[i].l2Norm()));
    std::size_t begin = 0;
    // sum of TAC and sum of segment length
    // TAC means "total absolute curvature"
    float tac_sum = 0.f, len_sum = 0.f;
    for (std::size_t end = 1; end < samples.size() - 1; end++) {
        auto len = samples[end + 1]->distance(*samples[end - 1]);
        auto tac = std::fabsf(kappa[end - 1]) * len;
        if ((tac_sum + tac > crv_thresh || len_sum + len > len_thresh)) {
            auto seg = new Segment{ this, static_cast<int>(begin), static_cast<int>(end + 1) };
            segments.emplace_back(seg);
            shape->segments.emplace_back(seg);
            seg->table = new SegmentPointLookupTable{ seg };
            tac_sum = len_sum = 0.f;
            begin = end;
        } else {
            tac_sum += tac;
            len_sum += len;
        }
    }
    // finalize the last segment (if unfinished)
    if (segments.empty() || segments.back()->end != samples.size()) {
        auto seg = new Segment{ this, static_cast<int>(begin), static_cast<int>(samples.size()) };
        segments.emplace_back(seg);
        shape->segments.emplace_back(seg);
        seg->table = new SegmentPointLookupTable{ seg };
    }
}

void Curve::decomposeGlobal(float, float penalty = 625) {
	penalty = 1.f;
	float *f = new float[samples.size()];
	int *ll_f = new int[samples.size()];
	std::vector<Vector> r;
	r.resize(samples.size());
	for (std::size_t i = 1; i < samples.size(); i++)
		r[i] = (*samples[i] - *samples[i - 1]);
	// the second derivative
	std::vector<Vector> rr;
	rr.resize(samples.size());
	for (std::size_t i = 2; i < samples.size(); i++)
		rr[i] = (r[i] - r[i - 1]);

		f[0] = 0;
		ll_f[0] = 0;
		for (int j = 1; j < samples.size(); j++) {
			f[j] = std::numeric_limits<float>::max();
			for (int k = 0; k < j; k++) {
				Vector mean = 0.f;
				float distqrs = 0;
				float ratio = 0;
				for (int l = k + 1; l <= j; l++)
					mean += *samples[j];
				mean /= (j - k);
				for (int l = k + 1; l <= j; l++)
				{
					distqrs += (*samples[l] - mean).sq();
					ratio += (*samples[l] - *samples[l - 1]).l2Norm();
				}

				ratio /= (*samples[j] - *samples[k]).l2Norm();
				distqrs /= (float)(j - k);

				int curvature_idx = k;
				if (k < 2)
					curvature_idx = 2;
				float curvature = rr[curvature_idx].sq() / cubic(sqrtf(r[curvature_idx].sq() + 1));//r[curvature_idx].cross(rr[curvature_idx]).l2Norm() / cubic(r[curvature_idx].l2Norm()); 
				if (curvature == 0)
					curvature = 1e-6;
				if (f[k] + distqrs + penalty / curvature < f[j])
				{
					f[j] = f[k] + distqrs + penalty / curvature;
					ll_f[j] = k;
				}

			}
		}
		std::vector<Segment* > currsegs;
		int j = samples.size() - 1;

		while (j > 0) {
			currsegs.push_back(new Segment(this, ll_f[j], j + 1));
			j = ll_f[j];
		}

		for (Segment *seg : currsegs)
		{
			segments.push_back(seg);
			shape->segments.push_back(seg);
			seg->table = new SegmentPointLookupTable{ seg };
		}

	delete[] f;
	delete[] ll_f;
}

void Curve::decomposeByRatio(float ratio_thresh, float len_thresh) {
    auto len_sum = 0.f;
    std::size_t begin = 0;
    for (std::size_t end = 1; end < samples.size() - 1; end++) {
        auto len = samples[end + 1]->distance(*samples[end]);
        if (len_sum + len > ratio_thresh * samples[end + 1]->distance(*samples[begin]) || len_sum + len > len_thresh) {
            auto seg = new Segment{ this, static_cast<int>(begin), static_cast<int>(end + 1) };
            segments.emplace_back(seg);
            shape->segments.emplace_back(seg);
            seg->table = new SegmentPointLookupTable{ seg };
            if (len_sum + len > len_thresh)
                seg->stop_by_length = true;
            len_sum = 0.f;
            begin = end;
        } else {
            len_sum += len;
        }
    }
    if (segments.empty() || segments.back()->end < samples.size()) {
        auto seg = new Segment{ this, static_cast<int>(begin), static_cast<int>(samples.size()) };
        segments.emplace_back(seg);
        shape->segments.emplace_back(seg);
        seg->table = new SegmentPointLookupTable{ seg };
    }
}

Point::Point(Curve *crv, Vector v) : curve{ crv }, index{ static_cast<int>(crv->samples.size()) }, Vector{ v } {}

struct LshParam {
    // the number of hash functions
    int n_functions;//5
    // the number of hash buckets
    int n_buckets;//20
    // how many results we want in a single query
    int k;
    // the `w` parameter in LSH function
    // according to e2LSH, 4.0 is the best value in practice
    float w;
};

// LSH hash function
class LshFunc {
    Vector a_;
    float b_, w_;
public:
    
    LshFunc(Vector a, float b, float w) : a_{ a }, b_{ b }, w_{ w } {}

    inline float operator() (const Vector &x) const {
        return (x.dot(a_) + b_) / w_;
    }

    inline float operator() (Segment *seg) const {
        return (*this)(seg->centroid);
    }

    static std::vector<LshFunc> create(int n, float w) {
        std::random_device rd{};
        std::mt19937_64 engine{ rd() };
        std::uniform_real_distribution<float> uni_dist{ 0, w };
        std::normal_distribution<float> gauss_dist{ 0, 1. };
        std::vector<LshFunc> fn;
        for (int i = 0; i < n; i++)
            fn.emplace_back(Vector{ gauss_dist(engine), gauss_dist(engine), gauss_dist(engine) }, uni_dist(engine), w);
        return fn;
    }
};

class LshSegmentSet {
    // the hash bucket
    struct Bucket {
        // used for BFS, indicating whether this bucket is unvisited or visited
        int color = 0;
        // the indexes of this bucket in all hash functions
        std::vector<int> hash_codes;

        // the segments it contains
        std::vector<Segment*> segments;
        // all adjacent buckets, a adjacent bucket is defined as
        // the bucket that has similar hash value in some hash function
        std::vector<Bucket*> adjacents;
    };

    // since hash value of points are floating-point values
    // we need `Projector`, a helper class, to map the FP values into unsigned integers
    struct Projector {
        std::vector<std::vector<Bucket*>> buckets;
        std::vector<int> next, prev;
        int n_buckets;
        // minimal value of the range
        float min = std::numeric_limits<float>::max();
        // maximal value of the range
        float max = std::numeric_limits<float>::min();
        // the bucket width
        float step = 0.f;

        // map a FP hash value to an integer
        inline int operator() (float x) const {
            if (x - min < EPSILON<>)
                return 0;
            if (max - x < EPSILON<>)
                return n_buckets - 1;
            return static_cast<int>(std::floor((x - min) / step));
        }
    };

    Shape *shape_;
    LshParam param_;
    std::vector<Bucket*> all_buckets_;
    std::vector<LshFunc> lsh_funcs_;
    std::vector<Projector> projectors_;
    std::unordered_map<int, Bucket*> map_;
    int bfs_color_ = 0;
    int bucket_hit_counter_ = 0;

    void findNearestBucket(const Vector &s, std::queue<Bucket*> &q) {
        int hash_code_sum = 0;
        std::vector<int> hash_code;
        for (int i = 0; i < param_.n_functions; i++) {
            hash_code.emplace_back(projectors_[i](lsh_funcs_[i](s)));
            hash_code_sum += hash_code.back();
            hash_code_sum *= param_.n_buckets;
        }
        auto find_result = map_.find(hash_code_sum);
        if (find_result != map_.end()) {
            q.push(find_result->second);
            bucket_hit_counter_++;
        } else {
            std::vector<Bucket*> bins;
            for (int i = 0; i < param_.n_functions; i++) {
                int prev = i - 1;
                while (prev >= 0 && projectors_[i].buckets[prev].empty())
                    prev--;
                if (prev >= 0)
                    for (auto b : projectors_[i].buckets[prev])
                        bins.emplace_back(b);
                int next = i + 1;
                while (next < param_.n_buckets && projectors_[i].buckets[next].empty())
                    next++;
                if (next < param_.n_buckets)
                    for (auto b : projectors_[i].buckets[next])
                        bins.emplace_back(b);
            }
            for (auto b : bins) {
                q.push(b);
                b->color = bfs_color_ + 1;
            }
        }
    }
public:
    LshSegmentSet(Shape &shape, LshParam param)
        : shape_{ &shape }, param_{ param }, lsh_funcs_{ LshFunc::create(param.n_functions, param.w) } {
        std::vector<std::vector<float>> hash_values{ shape.segments.size() };
        projectors_.resize(param.n_functions);
        // compute hash values for all centroids
        for (std::size_t i = 0; i < shape.segments.size(); i++) {
            for (int j = 0; j < param.n_functions; j++) {
                auto x = lsh_funcs_[j](shape.segments[i]->centroid);
                hash_values[i].emplace_back(x);
                // update min and max value for each hash function
                if (x < projectors_[j].min)
                    projectors_[j].min = x;
                if (x > projectors_[j].max)
                    projectors_[j].max = x;
            }
        }
        // find the minimal and maximal value and
        // compute bucket width of each hash function
        for (int i = 0; i < param.n_functions; i++) {
            projectors_[i].buckets.resize(param.n_buckets);
            projectors_[i].n_buckets = param.n_buckets;
            projectors_[i].step = (projectors_[i].max - projectors_[i].min) / static_cast<float>(param.n_buckets);
        }
        std::vector<std::vector<int>> hash_codes;
        std::vector<int> hash_code_sums;
        // compute hash code for each centroid
        hash_codes.resize(hash_values.size());
        for (std::size_t i = 0; i < hash_values.size(); i++) {
            hash_code_sums.emplace_back(0);
            hash_codes[i].reserve(param.n_functions);
            for (int j = 0; j < param.n_functions; j++) {
                hash_codes[i].emplace_back(projectors_[j](hash_values[i][j]));
                hash_code_sums.back() += hash_codes[i].back();
                hash_code_sums.back() *= param.n_buckets;
            }
        }
        for (std::size_t i = 0; i < hash_values.size(); i++) {
            // append centroid to corresponding bucket
            // create if not exists
            auto find_result = map_.find(hash_code_sums[i]);
            if (find_result == map_.end()) {
                auto b = new Bucket;
                all_buckets_.emplace_back(b);
                b->hash_codes = hash_codes[i];
                map_[hash_code_sums[i]] = b;
                b->segments.emplace_back(shape.segments[i]);
                for (int j = 0; j < param.n_functions; j++) {
                    projectors_[j].buckets[hash_codes[i][j]].emplace_back(b);
                }
            } else {
                find_result->second->segments.emplace_back(shape.segments[i]);
            }
        }
        // find the next and previous bucket index for each hash function
        for (auto &proj : projectors_) {
            proj.prev.resize(param.n_buckets);
            proj.next.resize(param.n_buckets);
            for (int j = 0; j < param.n_buckets; j++) {
                proj.prev[j] = j - 1;
                while (proj.prev[j] >= 0 && proj.buckets[proj.prev[j]].empty())
                    proj.prev[j]--;
                proj.next[j] = j + 1;
                while (proj.next[j] < param.n_buckets && proj.buckets[proj.next[j]].empty())
                    proj.next[j]++;
            }
        }
        // build adjacent info
        for (auto proj : projectors_) {
            std::size_t i = 0, next;
            while (i < param.n_buckets) {
                next = i + 1;
                while (next < param.n_buckets && proj.buckets[next].empty())
                    next++;
                if (next == param.n_buckets)
                    break;
                for (auto b1 : proj.buckets[i])
                    for (auto b2 : proj.buckets[next]) {
                        b1->adjacents.emplace_back(b2);
                        b2->adjacents.emplace_back(b1);
                    }
                i = next;
            }
        }
        // compress adjacent list
        for (auto bin : all_buckets_) {
            std::vector<std::pair<int, Bucket*>> adjacents;
            for (auto adj : bin->adjacents) {
                int common = 0;
                for (int i = 0; i < param.n_functions; i++)
                    if (projectors_[i].next[bin->hash_codes[i]] == adj->hash_codes[i])
                        common++;
                    else if (projectors_[i].prev[bin->hash_codes[i]] == adj->hash_codes[i])
                        common++;
                assert(common > 0);
                adjacents.emplace_back(common, adj);
            }
            std::sort(begin(adjacents), end(adjacents), greaterFirst<int, Bucket*>);
            auto n = std::min(std::size_t{ 20 }, adjacents.size());
            bin->adjacents.clear();
            for (std::size_t i = 0; i < n; i++)
                bin->adjacents.emplace_back(adjacents[i].second);
            bin->adjacents.shrink_to_fit();
        }
    }

    // get how many times we hit the bucket
    int get_bucket_hit_count() const {
        return bucket_hit_counter_;
    }

    QueryResult queryApproximateNeighbours(const Vector &s, std::size_t k) {
        // the `i`-th element of `seg_of_crv` represents the nearest segment of `i`-th curve
        std::vector<Segment*> seg_of_crv{ shape_->curves.size(), nullptr };
        // how many segments we've found?
        std::size_t found = 0;
        // the queue we used for BFS
        std::queue<Bucket*> q;
        findNearestBucket(s, q);
        while (found < k) {
            if (q.empty())
                throw std::runtime_error("empty queue");
            auto bin = q.front();
            q.pop();
            bin->color = bfs_color_ + 2; // mark `bin` as visited
            for (auto seg : bin->segments) {
                auto crv_idx = seg->curve->index;
                if (seg_of_crv[crv_idx] == nullptr) {
                    found++;
                    seg_of_crv[crv_idx] = seg;
                } else if (seg_of_crv[crv_idx]->centroid.distance(s) < seg->centroid.distance(s))
                    seg_of_crv[crv_idx] = seg;
            }
            for (auto adj : bin->adjacents)
                if (adj->color <= bfs_color_) {
                    adj->color = bfs_color_ + 1; // mark `bin` as seen
                    q.push(adj);
                }
        }
        // change the color
        bfs_color_ += 3;
        // collect query result from `seg_of_crv`
        QueryResult res;
        res.reserve(found);
        for (auto seg : seg_of_crv)
            if (seg) {
                auto pt = seg->table->nearest(s);
                //auto pt = seg->findNearestPoint(s);
                res.emplace_back(pt->distance(s), pt);
            }
        return res;
    }
};

QueryResult findKNearestNeighbours(const Shape &shape, const Vector &s, std::size_t k) {
    QueryResult heap{ k, { std::numeric_limits<float>::infinity(), nullptr } };
    for (auto crv : shape.curves) {
        auto t = *std::min_element(begin(crv->samples), end(crv->samples), [s](Point *lhs, Point *rhs) {
            return lhs->distance(s) < rhs->distance(s);
        });
        if (t->distance(s) < heap.front().first) {
            std::pop_heap(begin(heap), end(heap));
            heap.back() = { t->distance(s), t };
            std::push_heap(begin(heap), end(heap));
        }
    }
    std::sort_heap(begin(heap), end(heap));
    return heap;
}

float evaluateError(const QueryResult &knn, const QueryResult &ann) {
    assert(knn.size() <= ann.size());
    assert(std::is_sorted(begin(knn), end(knn), lessFirst<float, Point*>));
    assert(std::is_sorted(begin(ann), end(ann), lessFirst<float, Point*>));
    float err_sum = 0.f;
    for (std::size_t i = 0; i < knn.size(); i++)
        err_sum += std::fabsf(knn[i].first - ann[i].first);
    return err_sum / static_cast<float>(knn.size());
}

#include <chrono> 
#include <iostream>

std::ostream &operator<< (std::ostream &out, const Vector &v) {
    return out << '(' << v.x << ',' << ' ' << v.y << ',' << ' ' << v.z << ')';
}

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_, end_;
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedSeconds() const {
        std::chrono::duration<double> dur = end_ - start_;
        return dur.count();
    }
};

// Performance Counter
class Performance {
    int n_samples_ = 0;
    double time_sum_ = 0.;
    float error_sum_ = 0.f;
public:
    inline void log(double time, float error) {
        n_samples_++;
        time_sum_ += time;
        error_sum_ += error;
    }

    inline double totalTime() const {
        return time_sum_;
    }

    inline double averageTime() const {
        return time_sum_ / static_cast<double>(n_samples_);
    }

    inline float averageError() const {
        return error_sum_ / static_cast<float>(n_samples_);
    }
};

void benchmark() {
    const int k = 100;
    Shape shape;
    shape.load(
        "D:\\flow_data\\aneurysm.obj",
        //"D:\\flow_data\\aneurysm.obj",
        DecompositionPolicy::Curvature,
        { 40*PI<>, 10.f }
    );

    int cnt = 0;
    for (auto seg : shape.segments) {
        float dist_sum = 0.f;
        for (auto pt : seg) {
            dist_sum += pt->distance(seg->centroid);
        }
        //std::cout << "Segment #" << ++cnt << " deviation; " << dist_sum << ' ' << seg->stop_by_length <<  '\n';
    }

    LshSegmentSet seg_set{ shape,{ 5, 20, k, 4.f } };
    Timer stopwatch;
    Performance pfm;
    int counter = 0;
    for (auto pt : shape.points) {
        std::cout << "Query #" << ++counter << ' ' << *pt << '\n';
        stopwatch.start();
        auto ground_truth = findKNearestNeighbours(shape, *pt, k);
        stopwatch.stop();
        std::cout << "KNN Time: " << stopwatch.elapsedSeconds() << "s elpased.\n";
        stopwatch.start();
        auto res = seg_set.queryApproximateNeighbours(*pt, k);
        //auto res = shape.find(*pt, k);
        std::sort(begin(res), end(res), lessFirst<float, Point*>);
        stopwatch.stop();
        auto ann_time = stopwatch.elapsedSeconds();
        std::cout << "ANN Time: " << ann_time << "s elpased.\n";
        auto ann_error = evaluateError(ground_truth, res);
        std::cout << "Average Error: " << ann_error << '\n';
        pfm.log(ann_time, ann_error);
        if (counter == 1000)
            break;
    }
    std::cout << "Bucket hit " << seg_set.get_bucket_hit_count() << " times.\n";
    std::cout << "Avg. Time: " << pfm.averageTime() << '\n';
    std::cout << "Avg. Error: " << pfm.averageError() << '\n';
    std::cout << "Total Time: " << pfm.totalTime() << '\n';
    int some = getchar();
}

std::vector<Curve*> loadQueryCurves(const char *filename) {
    FILE *fp = nullptr;
    if (fopen_s(&fp, filename, "r"))
        return {};
    std::vector<Curve*> crvs;
    int c;
    int idx, n, loop;
    while (fscanf_s(fp, "id=%d,n=%d,loop=%d", &idx, &n, &loop) != EOF) {
        crvs.emplace_back(new Curve(nullptr, idx));
        for (int i = 0; i < n; i++) {
            Vector v;
            fscanf_s(fp, "%f %f %f", &v.x, &v.y, &v.z);
            auto pt = new Point(crvs.back(), v);
            crvs.back()->samples.emplace_back(pt);
        }
        while ((c = fgetc(fp)) != '#');
        fgetc(fp);
    }
    return crvs;
}

std::vector<float> computeSimilarity(const Vector &v, const std::vector<Point*> &pts) {
    std::vector<float> sim;
	return std::vector<float>();
}

int main(int argc, char **argv) {
    const int k = 3;
    Shape shape;
    shape.load(
        //"C:\\Users\\IRCVIS\\Documents\\Tencent Files\\772103636\\FileRecv\\test3.obj",
        "D:\\flow_data\\GL3D_Xfieldramp_inter_0437_cop.obj",
        //"D:\\flow_data\\aneurysm.obj",
        DecompositionPolicy::Curvature,
        { PI<> * 2, 10.f }
    );

    LshSegmentSet seg_set{ shape,{ 5, 20, k, 4.f } };
    Timer stopwatch;
    Performance pfm;
    int counter = 0;
    for (auto pt : shape.points) {
        std::cout << "Query #" << ++counter << ' ' << *pt << '\n';
        stopwatch.start();
        auto ground_truth = findKNearestNeighbours(shape, *pt, k);
        stopwatch.stop();
        std::cout << "KNN Time: " << stopwatch.elapsedSeconds() << "s elpased.\n";
        stopwatch.start();
        auto res = seg_set.queryApproximateNeighbours(*pt, k);
        //auto res = shape.find(*pt, k);
        std::sort(begin(res), end(res), lessFirst<float, Point*>);
        stopwatch.stop();
        auto ann_time = stopwatch.elapsedSeconds();
        std::cout << "ANN Time: " << ann_time << "s elpased.\n";
        auto ann_error = evaluateError(ground_truth, res);
        std::cout << "Average Error: " << ann_error << '\n';
        pfm.log(ann_time, ann_error);
        if (counter == 1000)
            break;
    }
    std::cout << "Bucket hit " << seg_set.get_bucket_hit_count() << " times.\n";
    std::cout << "Avg. Time: " << pfm.averageTime() << '\n';
    std::cout << "Avg. Error: " << pfm.averageError() << '\n';
	std::cout << "Segments decomposed: " << shape.segments.size() << '\n';
	std::cout << "Total Time: " << pfm.totalTime() << '\n';
 //   int some = getchar();
}

Point *Segment::findNearestPoint(const Vector &s) {
    return *std::min_element(::begin(this), ::end(this), [s](Point *p1, Point *p2) {
        return p1->distance(s) < p2->distance(s);
    });
}

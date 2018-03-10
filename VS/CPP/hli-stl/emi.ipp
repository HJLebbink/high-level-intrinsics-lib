#pragma once

#include <algorithm>	// std::min, max_element
#include <limits>		// std::numeric_limits
#include <iostream>		// std::cout

#include <cmath>	//log2 lgamma
#include <tuple>
#include <vector>
#include <map>


#include "emmintrin.h"  // sse
#include "pmmintrin.h"  // sse3
#include "tmmintrin.h"  // ssse3
#include "smmintrin.h"  // sse4.1
#include "nmmintrin.h"  // sse4.2
//#include "immintrin.h"  // avx, avx2, avx512, FP16C, KNCNI, FMA
//#include "ammintrin.h"  // AMD-specific intrinsics

#include "tools.ipp"
#include "timing.ipp"

namespace hli
{
	namespace priv
	{
		/*
		static double emi(final List<Integer> A_array, final List<Integer> B_array)
		{
			final int N = A_array.size();
			assert(A_array.size() == B_array.size());

			final Map<Integer, Integer> f_a = freq(A_array);
			final Map<Integer, Integer> f_b = freq(B_array);

			//number of unique elements in A_array
			final int R = f_a.keySet().size();
			//number of unique elements in B_array
			final int C = f_b.keySet().size();

			//System.out.println("R="+R +"; C="+C);

			//freq of elements A_array (R)
			final List<Integer> a = new ArrayList<Integer>(f_a.values());
			//freq of elements in B_array (C)
			final List<Integer> b = new ArrayList<Integer>(f_b.values());

			final int s1 = (Math.max(Collections.max(a), Collections.max(b)) + 1);
			final List<Double> nijs = new ArrayList<Double>(s1);

			nijs.add(1.); // Stops divide by zero warnings. As its not used, no issue.
			for (int i = 1; i<s1; ++i) nijs.add((double)i);


			final List<Double> log_a = new ArrayList<Double>(R);
			final List<Double> gln_a = new ArrayList<Double>(R);
			final List<Double> gln_Na = new ArrayList<Double>(R);
			for (int i = 0; i<R; ++i)
			{
				final int a_val = a.get(i);
				log_a.add(log(a_val));
				gln_a.add(Gamma.logGamma(a_val + 1));
				gln_Na.add(Gamma.logGamma(N - a_val + 1));
			}

			final List<Double> log_b = new ArrayList<Double>(C);
			final List<Double> gln_b = new ArrayList<Double>(C);
			final List<Double> gln_Nb = new ArrayList<Double>(C);
			for (int i = 0; i<C; ++i)
			{
				final int b_val = b.get(i);
				log_b.add(log(b_val));
				gln_b.add(Gamma.logGamma(b_val + 1));
				gln_Nb.add(Gamma.logGamma(N - b_val + 1));
			}

			final List<Double> term1 = new ArrayList<Double>(s1);
			final List<Double> log_nijs = new ArrayList<Double>(s1);
			final List<Double> log_Nnij = new ArrayList<Double>(s1);
			final List<Double> gln_nijs = new ArrayList<Double>(s1);
			for (int i = 0; i<s1; ++i)
			{
				final double nijs_val = nijs.get(i);
				term1.add(nijs_val / N);
				log_nijs.add(log(nijs_val));
				log_Nnij.add(log(N * nijs_val));
				gln_nijs.add(Gamma.logGamma(nijs_val + 1));
			}

			final double gln_N = Gamma.logGamma(N + 1);

			double emi = 0;
			for (int i = 0; i<R; ++i) for (int j = 0; j<C; ++j)
			{
				final double log_ab_outer = log_a.get(i) + log_b.get(j);
				final int ai = a.get(i);
				final int bj = b.get(j);
				final int start_i = Math.max(ai - N + bj, 1);
				final int end_i = Math.min(ai, bj) + 1;

				for (int nij = start_i; nij<end_i; ++nij)
				{
					final double term2 = log_Nnij.get(nij) - log_ab_outer;
					final double gln = (
						gln_a.get(i) + gln_b.get(j)
						+ gln_Na.get(i) + gln_Nb.get(j)
						- gln_N - gln_nijs.get(nij)
						- Gamma.logGamma(ai - nij + 1)
						- Gamma.logGamma(bj - nij + 1)
						- Gamma.logGamma(N - ai - bj + nij + 1));
					final double term3 = FastMath.exp(gln);
					emi += (term1.get(nij) * term2 * term3);
				}
			}
			return emi;
		}
		*/

		// Visit http://www.johndcook.com/stand_alone_code.html for the source of this code and more like it.

		#include <sstream>
		#include <iostream>
		#include <stdexcept>

		double Gamma(double x); // forward decl
		// Note that the functions Gamma and LogGamma are mutually dependent.

		inline double LogGamma(double x)
		{
			if (x <= 0.0)
			{
				std::stringstream os;
				os << "Invalid input argument " << x << ". Argument must be positive.";
				throw std::invalid_argument(os.str());
			}

			if (x < 12.0)
			{
				return log(fabs(Gamma(x)));
			}

			// Abramowitz and Stegun 6.1.41
			// Asymptotic series should be good to at least 11 or 12 figures
			// For error analysis, see Whittiker and Watson
			// A Course in Modern Analysis (1927), page 252

			static const double c[8] =
			{
				1.0 / 12.0,
				-1.0 / 360.0,
				1.0 / 1260.0,
				-1.0 / 1680.0,
				1.0 / 1188.0,
				-691.0 / 360360.0,
				1.0 / 156.0,
				-3617.0 / 122400.0
			};
			double z = 1.0 / (x*x);
			double sum = c[7];
			for (int i = 6; i >= 0; i--)
			{
				sum *= z;
				sum += c[i];
			}
			double series = sum / x;

			static const double halfLogTwoPi = 0.91893853320467274178032973640562;
			double logGamma = (x - 0.5)*log(x) - x + halfLogTwoPi + series;
			return logGamma;
		}

		inline double Gamma(double x)
		{
			if (x <= 0.0)
			{
				std::stringstream os;
				os << "Invalid input argument " << x << ". Argument must be positive.";
				throw std::invalid_argument(os.str());
			}

			// Split the function domain into three intervals:
			// (0, 0.001), [0.001, 12), and (12, infinity)

			///////////////////////////////////////////////////////////////////////////
			// First interval: (0, 0.001)
			//
			// For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
			// So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
			// The relative error over this interval is less than 6e-7.

			const double gamma = 0.577215664901532860606512090; // Euler's gamma constant

			if (x < 0.001)
				return 1.0 / (x*(1.0 + gamma * x));

			///////////////////////////////////////////////////////////////////////////
			// Second interval: [0.001, 12)

			if (x < 12.0)
			{
				// The algorithm directly approximates gamma over (1,2) and uses
				// reduction identities to reduce other arguments to this interval.

				double y = x;
				int n = 0;
				bool arg_was_less_than_one = (y < 1.0);

				// Add or subtract integers as necessary to bring y into (1,2)
				// Will correct for this below
				if (arg_was_less_than_one)
				{
					y += 1.0;
				}
				else
				{
					n = static_cast<int> (floor(y)) - 1;  // will use n later
					y -= n;
				}

				// numerator coefficients for approximation over the interval (1,2)
				static const double p[] =
				{
					-1.71618513886549492533811E+0,
					2.47656508055759199108314E+1,
					-3.79804256470945635097577E+2,
					6.29331155312818442661052E+2,
					8.66966202790413211295064E+2,
					-3.14512729688483675254357E+4,
					-3.61444134186911729807069E+4,
					6.64561438202405440627855E+4
				};

				// denominator coefficients for approximation over the interval (1,2)
				static const double q[] =
				{
					-3.08402300119738975254353E+1,
					3.15350626979604161529144E+2,
					-1.01515636749021914166146E+3,
					-3.10777167157231109440444E+3,
					2.25381184209801510330112E+4,
					4.75584627752788110767815E+3,
					-1.34659959864969306392456E+5,
					-1.15132259675553483497211E+5
				};

				double num = 0.0;
				double den = 1.0;
				int i;

				double z = y - 1;
				for (i = 0; i < 8; i++)
				{
					num = (num + p[i])*z;
					den = den * z + q[i];
				}
				double result = num / den + 1.0;

				// Apply correction if argument was not initially in (1,2)
				if (arg_was_less_than_one)
				{
					// Use identity gamma(z) = gamma(z+1)/z
					// The variable "result" now holds gamma of the original y + 1
					// Thus we use y-1 to get back the orginal y.
					result /= (y - 1.0);
				}
				else
				{
					// Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
					for (i = 0; i < n; i++)
						result *= y++;
				}

				return result;
			}

			///////////////////////////////////////////////////////////////////////////
			// Third interval: [12, infinity)

			if (x > 171.624)
			{
				// Correct answer too large to display. Force +infinity.
				double temp = DBL_MAX;
				return temp * 2.0;
			}

			return exp(LogGamma(x));
		}

		inline double lgamma_local(double x)
		{
			//return std::lgamma(x);
			//return lgamma(x);
			return LogGamma(x);
		}


		inline double log_local(double x)
		{
			return log(x);
			//return log2(x);
		}

		inline int max_element(const std::vector<int>& v)
		{
			return *std::max_element(v.begin(), v.end());
		}

		template <typename T>
		inline std::map<T, int> calc_freq(const std::vector<T>& data)
		{
			int n_elements = static_cast<int>(data.size());
			std::map<T, int> m;
			{
				for (int i = 0; i < n_elements; ++i)
				{
					T v = data[i];
					//std::cout << "i="<< i << "; v=" << v << "; m[v]=" << m[v] << std::endl;
					m[v] = m[v] + 1;
				}
			}
			return m;
		}

		template <typename T>
		inline std::vector<int> get_values(const std::map<T, int>& m) {
			std::vector<int> result;
			for (auto it = m.begin(); it != m.end(); ++it)
			{
				result.push_back(it->second);
			}
			return result;
		}

		template <typename T>
		inline double emi_method0(
			const std::map<T, int>& freq_a,
			const std::map<T, int>& freq_b,
			const int n_elements)
		{
			std::vector<int>a = get_values(freq_a);
			std::vector<int>b = get_values(freq_b);

			int N = n_elements;
			int R = static_cast<int>(a.size());
			int C = static_cast<int>(b.size());

			//std::cout << "N=" << N << "; R=" << R << "; C=" << C << std::endl;

			auto log_a = std::vector<double>(R);
			auto gln_a = std::vector<double>(R);
			auto gln_Na = std::vector<double>(R);
			for (int i = 0; i < R; ++i)
			{
				int ai = a[i];
				//std::cout << "i=" << i << "; ai=" << ai << std::endl;
				log_a[i] = log_local(ai);
				gln_a[i] = lgamma_local(ai + 1);
				gln_Na[i] = lgamma_local(N - ai + 1);
			}

			auto log_b = std::vector<double>(C);
			auto gln_b = std::vector<double>(C);
			auto gln_Nb = std::vector<double>(C);
			for (int j = 0; j < C; ++j)
			{
				int bj = b[j];
				//std::cout << "j=" << j << "; bj=" << bj << std::endl;
				log_b[j] = log_local(bj);
				gln_b[j] = lgamma_local(bj + 1);
				gln_Nb[j] = lgamma_local(N - bj + 1);
			}

			int nij_size = std::max(max_element(a), max_element(b)) + 1;
			//std::cout << "nij_size=" << nij_size << std::endl;

			auto nijs = std::vector<double>(nij_size);
			nijs[0] = 1; // Stops divide by zero warnings. As its not used, no issue.
			for (int i = 1; i < nij_size; ++i) nijs[i] = static_cast<double>(i);

			auto term1 = std::vector<double>(nij_size);
			auto log_nijs = std::vector<double>(nij_size);
			auto log_Nnij = std::vector<double>(nij_size);
			auto gln_nijs = std::vector<double>(nij_size);
			for (int nij = 0; nij < nij_size; ++nij)
			{
				double nijs_val = nijs[nij];
				term1[nij] = nijs_val / N;
				log_nijs[nij] = log_local(nijs_val);
				log_Nnij[nij] = log_local(N * nijs_val);
				gln_nijs[nij] = lgamma_local(nijs_val + 1);
			}

			double gln_N = lgamma_local(N + 1);
			double emi = 0;

			for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j)
			{
				double log_ab_outer = log_a[i] + log_b[j];
				int ai = a[i];
				int bj = b[j];
				int start_i = std::max(ai - N + bj, 1);
				int end_i = std::min(ai, bj) + 1;
				//std::cout << "start_i=" << start_i << "; end_i=" << end_i << "; ai=" << ai << "; bj="<< bj << std::endl;

				for (int nij = start_i; nij < end_i; ++nij)
				{
					double term2 = log_Nnij[nij] - log_ab_outer;
					double gln = gln_a[i] + gln_b[j]
						+ gln_Na[i] + gln_Nb[j]
						- gln_N - gln_nijs[nij]
						- lgamma_local(ai - nij + 1)
						- lgamma_local(bj - nij + 1)
						- lgamma_local(N - ai - bj + nij + 1);
					double term3 = exp(gln);

					//std::cout << "term1=" << term1[nij] << "; term2=" << term2 << "; term3=" << term3 << std::endl;
					emi += (term1[nij] * term2 * term3);
				}
			}
			return emi;
		}
	}

	inline double entropy(
		const std::vector<int>& A)
	{
		int N = static_cast<int>(A.size());
		auto freq = priv::calc_freq(A);
		double h = 0;
		for (auto const& v : freq)
		{
			//std::cout << "freq:" << v.second << std::endl;
			double p = static_cast<double>(v.second) / N;
			h += p * priv::log_local(p);
		}
		return -h;
	}

	inline double mi(
		const std::vector<int>& A,
		const std::vector<int>& B)
	{
		int N = static_cast<int>(A.size());
		std::vector<int> AB = std::vector<int>(N);
		for (int i = 0; i < N; ++i) AB[i] = A[i] << 16 | B[i];

		return (entropy(A) + entropy(B)) - entropy(AB);
	}

	inline double emi(
		const std::vector<int>& A,
		const std::vector<int>& B)
	{
		int N = static_cast<int>(A.size());
		return priv::emi_method0(priv::calc_freq(A), priv::calc_freq(B), N);
	}

	inline double ami(
		const std::vector<int>& A,
		const std::vector<int>& B)
	{
		double hA = entropy(A);
		double hB = entropy(B);
		double emiAB = emi(A, B);
		double miAB = mi(A, B);
		return (miAB - emiAB) / (std::max(hA, hB) - emiAB);
	}


	namespace test
	{
		void __mm_emi_epu8_methode0_test_0()
		{
			auto A = std::vector<int>{ 0, 1, 1 };
			auto B = std::vector<int>{ 0, 0, 1 };

			std::cout << " mi([0,1,1],[0,0,1])=" << mi(A, B) << std::endl;
			std::cout << "emi([0,1,1],[0,0,1])=" << emi(A, B) << std::endl;
			std::cout << "ami([0,1,1],[0,0,1])=" << ami(A, B) << std::endl;
		}

		// Test speed of mutual information 8-bits unsigned integers no missing values
		void __mm_emi_epu8_methode0_test_1(
			const int n_elements,
			const int nExperiments)
		{
			const int N_BITS_A = 2;
			const int N_BITS_B = 2;
			const int MASK_A = (1 << N_BITS_A)-1;
			const int MASK_B = (1 << N_BITS_B)-1;

			auto A = std::vector<int>(n_elements);
			auto B = std::vector<int>(n_elements);

			for (int i = 0; i < n_elements; ++i)
			{
				A[i] = rand() & MASK_A;
				B[i] = rand() & MASK_B;
			}

			auto freq_A = priv::calc_freq(A);
			auto freq_B = priv::calc_freq(B);

			double min0 = std::numeric_limits<double>::max();

			double result0;

			for (int i = 0; i < nExperiments; ++i)
			{
				reset_and_start_timer();
				result0 = priv::emi_method0(freq_A, freq_B, n_elements);
				min0 = std::min(min0, get_elapsed_kcycles());
			}
			printf("[emi_method0]: %2.5f Kcycles; %0.14f\n", min0, result0);
		}
	}
}
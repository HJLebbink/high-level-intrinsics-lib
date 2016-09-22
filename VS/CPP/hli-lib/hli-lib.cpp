#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#if !defined(NOMINMAX)
#define NOMINMAX 1 
#endif
#if !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#endif

#include "..\hli-stl\_mm_corr_epu8.h"
#include "..\hli-stl\_mm_corr_pd.h"
#include "..\hli-stl\_mm_covar_epu8.h"
#include "..\hli-stl\_mm_entropy_epu8.h"
#include "..\hli-stl\_mm_hadd_epi64.h"
#include "..\hli-stl\_mm_hadd_epu8.h"
#include "..\hli-stl\_mm_mi_epu8.h"
#include "..\hli-stl\_mm_permute_array.h"
#include "..\hli-stl\_mm_rand_si128.h"
#include "..\hli-stl\_mm_rescale_epu16.h"
#include "..\hli-stl\_mm_variance_epu8.h"

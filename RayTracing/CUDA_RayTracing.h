// CUDA_RayTracing.h : main header file for the CUDA_RayTracing application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CCUDA_RayTracingApp:
// See CUDA_RayTracing.cpp for the implementation of this class
//

class CCUDA_RayTracingApp : public CWinApp
{
public:
	CCUDA_RayTracingApp();


// Overrides
public:
	virtual BOOL InitInstance();

// Implementation
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CCUDA_RayTracingApp theApp;
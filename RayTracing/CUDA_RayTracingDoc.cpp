// CUDA_RayTracingDoc.cpp : implementation of the CCUDA_RayTracingDoc class
//

#include "stdafx.h"
#include "CUDA_RayTracing.h"

#include "CUDA_RayTracingDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CCUDA_RayTracingDoc

IMPLEMENT_DYNCREATE(CCUDA_RayTracingDoc, CDocument)

BEGIN_MESSAGE_MAP(CCUDA_RayTracingDoc, CDocument)
END_MESSAGE_MAP()


// CCUDA_RayTracingDoc construction/destruction

CCUDA_RayTracingDoc::CCUDA_RayTracingDoc()
{
	// TODO: add one-time construction code here

}

CCUDA_RayTracingDoc::~CCUDA_RayTracingDoc()
{
}

BOOL CCUDA_RayTracingDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}




// CCUDA_RayTracingDoc serialization

void CCUDA_RayTracingDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}


// CCUDA_RayTracingDoc diagnostics

#ifdef _DEBUG
void CCUDA_RayTracingDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CCUDA_RayTracingDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CCUDA_RayTracingDoc commands

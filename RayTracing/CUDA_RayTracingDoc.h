// CUDA_RayTracingDoc.h : interface of the CCUDA_RayTracingDoc class
//


#pragma once


class CCUDA_RayTracingDoc : public CDocument
{
protected: // create from serialization only
	CCUDA_RayTracingDoc();
	DECLARE_DYNCREATE(CCUDA_RayTracingDoc)

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);

// Implementation
public:
	virtual ~CCUDA_RayTracingDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};



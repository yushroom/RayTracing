// MySplitterWnd.cpp : implementation file
//

#include "stdafx.h"
#include "MySplitterWnd.h"


// CMySplitterWnd

IMPLEMENT_DYNAMIC(CMySplitterWnd, CWnd)

CMySplitterWnd::CMySplitterWnd()
{

}

CMySplitterWnd::~CMySplitterWnd()
{
}


BEGIN_MESSAGE_MAP(CMySplitterWnd, CSplitterWnd)
	ON_WM_LBUTTONDOWN()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()



// CMySplitterWnd message handlers



void CMySplitterWnd::OnLButtonDown(UINT nFlags, CPoint point)
{
	// Do nothing in the LButton Down message to disable modifing the splitter Bar

//	CSplitterWnd::OnLButtonDown(nFlags, point);
}

void CMySplitterWnd::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

//	CSplitterWnd::OnMouseMove(nFlags, point);
}

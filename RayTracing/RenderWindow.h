/*
 *	FileName:	RenderWindow.h
 *
 *	Programmer:	Jiayin Cao
 */

#pragma once

#include "define.h"

////////////////////////////////////////////////////////////
// CRenderWindow dialog
class CRenderWindow : public CDialog
{
	DECLARE_DYNAMIC(CRenderWindow)

public:
	//constructor
	CRenderWindow(CWnd* pParent = NULL);
	//destructor
	virtual ~CRenderWindow();
	
	//show the image
	void	ShowImage();

	//set image resolution
	void	SetImageResolution( int w , int h );

	//get the resolution of the image
	int		GetImageWidth();
	int		GetImageHeight();

	//change image resolution
	void	ChangeImageResolution();

	//release the memory
	void	ReleaseMemory();

	//draw the image window
	void	ShowImageWindow( bool show = true );

	//save the image
	void	SaveImage(LPCWSTR saveFileAddName , COLORREF* imageData , int width , int height);

	//get the buffer of the image
	COLORREF*	GetBuffer();

	//present buffer
	void	PresentBuffer();

	//clear the buffer
	void	ClearBuffer();

	//set cpu ray tracing
	void	SetCPURayTrace( bool enable );

	//refresh the window
	void	Refresh();

	//stop RT
	void	StopRayTracing();

	//enable tool bar
	void	EnableToolBar( BOOL enable );

// Dialog Data
	enum { IDD = IDD_RENDERWINDOW };

//private field
private:
	//the tool bar
	CToolBar	m_ToolBar;

	//the height for the tool bar
	const int	m_ToolBarHeight;

	//the width of the current window
	int			m_iWidth;

	//the buffer for the image
	COLORREF*	m_pImageBuffer;
	COLORREF*	m_pBackBuffer;

	//the image size
	int			m_iCurImageWidth;
	int			m_iCurImageHeight;
	int			m_iDestImageWidth;
	int			m_iDestImageHeight;

	//the channel state
	bool	m_bRedChannelEnable;
	bool	m_bBlueChannelEnable;
	bool	m_bGreenChannelEnable;

	//whether cpu is doing ray tracing
	bool	m_bCPURayTracing;

//private method

	//initialize default value
	void	InitializeDefault();

/////////////////////////////////////////////////////////////////////////////
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()

//the message handler
public:
	virtual BOOL OnInitDialog();						//init the dialog
	afx_msg void OnPaint();								//paint the dialog
	afx_msg void OnSize(UINT nType, int cx, int cy);	//change the size of the dialog
	afx_msg void OnSaveImage();							//save the image
	afx_msg void OnRedChannel();
	afx_msg void OnBlueChannel();
	afx_msg void OnGreenChannel();
	virtual void OnCancel();
protected:
	virtual void OnOK();
public:
	afx_msg void OnClose();
};

/*
 *	FileName:	LoadingWindow.h
 *
 *	Programmer:	Jiayin Cao
 */

#pragma once

/////////////////////////////////////////////////////////////
class CLoadingWindow : public CDialog
{
	DECLARE_DYNAMIC(CLoadingWindow)

//public method
public:
	//constructor and destructor
	CLoadingWindow();
	virtual ~CLoadingWindow();

	//update progress dialog
	void	UpdateProgress( int progress );

	//update progress
	void	UpdateProgress();

//private field
private:
	//the buffer
	WCHAR	m_Str[1024];

protected:
	DECLARE_MESSAGE_MAP()
};



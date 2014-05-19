/*
 *	FileName:	Camera.h
 *
 *	Programmer:	Jiayin Cao
 *
 *	Description:	This is the camera
 */

#pragma once

#include <d3dx9.h>
#include "define.h"

/////////////////////////////////////////////////////////////////////
class Camera
{
//public method
public:
	Camera();
	~Camera();

	//Default value
	void	InitializeDefault();

	//Set Camera
	void	SetCamera( float angleV , float angleH , float radius , D3DXVECTOR3& target );

	//Update the Camera According to the cursor
	void	Update( POINT* p , bool rotate , int wheelPos , bool move , D3DXMATRIX& proj );

	//Update the view matrix
	void	UpdateMatrix();

	//Move the camera
	void	MoveCamera( POINT* p , D3DXMATRIX& proj );

	//Get ViewMatrix
	D3DXMATRIX&	GetViewMatrix();

	//Generate Ray
	void	GenerateRay( POINT* p , D3DXMATRIX& view , D3DXMATRIX& projection , D3DXVECTOR3& ori , D3DXVECTOR3& dir  );

	//Update Client Rect
	void	SetClientRect( int w , int h );

	//Get Eye position
	D3DXVECTOR3	GetEyePosition();

//private field
private:
	//the view matrix
	D3DXMATRIX	m_ViewMatrix;

	//eye point
	D3DXVECTOR3	m_vEyePosition;
	//target
	D3DXVECTOR3	m_vTarget;
	//up
	D3DXVECTOR3	m_vUp;

	//previous cursor position
	POINT	m_PreCursorPos;

	//the sphere coordinate property
	float	m_fAngleV;
	float	m_fAngleH;
	float	m_fRadius;

	//previous wheel position
	int		m_iPreWheelPos;

	//whether to reset plane
	bool	m_bResetPlane;
	//Normal of the plane
	D3DXPLANE	m_MovePlane;

	//the screen size
	int		m_iClientWidth;
	int		m_iClientHeight;

	//the matrix during moving
	D3DXMATRIX m_PreMatrix;

	//the intersected point
	D3DXVECTOR3	m_PreIntersectedPoint;

	//the minium radius
	float	m_fMinRadius;
};
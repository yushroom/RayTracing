/*
 *	FileName:	Camera.cpp
 *
 *	Programmer:	Jiayin Cao
 *
 *	Description:	This is the camera
 */

//include the header
#include "stdafx.h"
#include "Camera.h"

//constructor
Camera::Camera()
{
	//set default value
	InitializeDefault();
}

Camera::~Camera()
{
}

//Set Camera
void Camera::SetCamera( float angleV , float angleH , float radius , D3DXVECTOR3& target )
{
	m_fAngleV = angleV;
	m_fAngleH = angleH;
	m_fRadius = max( radius , m_fMinRadius );

	//set the target
	m_vTarget = target;

	//update the matrix
	UpdateMatrix();
}

//update the view matrix
void Camera::UpdateMatrix()
{
	//Update the camera
	m_vEyePosition.x = m_vTarget.x + m_fRadius * cos( m_fAngleH ) * sin( m_fAngleV );
	m_vEyePosition.y = m_vTarget.y + m_fRadius * sin( m_fAngleH );
	m_vEyePosition.z = m_vTarget.z + m_fRadius * cos( m_fAngleH ) * cos( m_fAngleV );

	//set up vector
	m_vUp = D3DXVECTOR3( 0.0f , 1.0f , 0.0f );

	//Update the view matrix
	D3DXMatrixLookAtLH( &m_ViewMatrix , &m_vEyePosition , &m_vTarget , &m_vUp );
}

//Default value
void Camera::InitializeDefault()
{
	//set default wheel position
	m_iPreWheelPos = DEFAULT_CURSOR_WHEEL_POS;

	m_fMinRadius = 2.0f;
}

//Update the Camera According to the cursor
void Camera::Update( POINT* p , bool rotate , int wheelPos , bool move , D3DXMATRIX& proj )
{
	//check if there is pointer available
	if( NULL == p )
		return;
	if( p->x < 0 || p->x > m_iClientWidth || p->y < 0 || p->y > m_iClientHeight )
		return;

	//Move the camera first
	if( move )
	{
		//move the camera first
		MoveCamera( p , proj );

		//don't reset plane next time
		m_bResetPlane = false;
	}
	else
		//reset plane
		m_bResetPlane = true;

	//Rotate the Camera
	if( rotate )
	{
		m_fAngleV -= ( m_PreCursorPos.x - p->x ) / 100.0f ;
		m_fAngleH -= ( m_PreCursorPos.y - p->y ) / 100.0f ;

		m_fAngleH = max( -(float)D3DX_PI / 2 + 0.01f , m_fAngleH );
		m_fAngleH = min( (float)D3DX_PI / 2 - 0.01f , m_fAngleH );
	}

	//update the cursor position
	m_PreCursorPos = *p;

	//Update the radius
	m_fRadius -= ( wheelPos - m_iPreWheelPos ) / 4.0f;

	//limit the minium value for radius
	m_fRadius = max( m_fMinRadius , m_fRadius );

	//update previous wheel position
	m_iPreWheelPos = wheelPos;

	//Update the matrix
	UpdateMatrix();
}

//Move the camera
void Camera::MoveCamera( POINT* p , D3DXMATRIX& proj )
{
	if( m_bResetPlane )
	{
		//reset the plane first
		D3DXVECTOR3 normal;
		D3DXVec3Normalize( &normal , &( m_vEyePosition - m_vTarget ) );

		m_MovePlane.a = normal.x;
		m_MovePlane.b = normal.y;
		m_MovePlane.c = normal.z;
		m_MovePlane.d = -1.0f * D3DXVec3Dot( &m_vTarget , &normal );

		//update the view matrix
		m_PreMatrix = m_ViewMatrix;
	}

	//Generate the ray
	D3DXVECTOR3	dir , ori , p2;
	GenerateRay( p , m_PreMatrix , proj , ori , dir );
	p2 = ori + 100000.0f * dir;

	//get the intersected point
	D3DXVECTOR3	intersected;
	D3DXPlaneIntersectLine( &intersected , &m_MovePlane , &ori , &p2 );

	//add offset to the target
	if( !m_bResetPlane )
		m_vTarget += m_PreIntersectedPoint - intersected;

	//update previous target
	m_PreIntersectedPoint = intersected;
}

//Generate Ray
void Camera::GenerateRay( POINT* p , D3DXMATRIX& view , D3DXMATRIX& projection , D3DXVECTOR3& ori , D3DXVECTOR3& dir )
{
	//the ray from eye to the pixel
	D3DXVECTOR3 ray;
	ray.x = ( ( ( 2.0f * p->x ) / m_iClientWidth ) - 1.0f ) / projection._11;
	ray.y = -( ( ( 2.0f * p->y ) / m_iClientHeight ) - 1.0f ) / projection._22;
	ray.z = 1.0f;

	//inverse the view matrix
	D3DXMATRIX invViewMatrix;
	D3DXMatrixInverse( &invViewMatrix , NULL , &view );

	//set the direction
	dir.x = invViewMatrix._11 * ray.x + invViewMatrix._21 * ray.y + invViewMatrix._31 * ray.z;
	dir.y = invViewMatrix._12 * ray.x + invViewMatrix._22 * ray.y + invViewMatrix._32 * ray.z;
	dir.z = invViewMatrix._13 * ray.x + invViewMatrix._23 * ray.y + invViewMatrix._33 * ray.z;

	//set the original point
	ori.x = invViewMatrix._41;
	ori.y = invViewMatrix._42;
	ori.z = invViewMatrix._43;
}

//Get ViewMatrix
D3DXMATRIX&	Camera::GetViewMatrix()
{
	return m_ViewMatrix;
}

//Update Client Rect
void Camera::SetClientRect( int w , int h )
{
	m_iClientWidth = w;
	m_iClientHeight = h;
}

//Get Eye position
D3DXVECTOR3 Camera::GetEyePosition()
{
	return m_vEyePosition;
}
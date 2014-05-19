/*
 *	FileName:	datatype.cpp
 *
 *	Programmer:	Jiayin Cao
 */

#include "datatype.h"
#include <math.h>
#include <windows.h>

//some helper functions
_float4	cross( const _float4& v1 , const _float4& v2 )
{
	_float4 r;
	r.x = v1.y * v2.z - v1.z * v2.y;
	r.y = v1.z * v2.x - v1.x * v2.z;
	r.z = v1.x * v2.y - v1.y * v2.x;
	r.w = 0.0f;

	return r;
}

//the dot product of the vectors
float dot( const _float4& v1 , const _float4& v2 )
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z ;
}

//default constructor
_float4::_float4()
{
	x = 0;
	y = 0;
	z = 0;
	w = 0;
}

_float4::_float4( float x , float y , float z , float w )
{
	this->x = x;
	this->y = y;
	this->z = z;
	this->w = w;
}

_float4::_float4( float* d )
{
	x = d[0];
	y = d[1];
	z = d[2];
	w = d[3];
}

//some operators
_float4 _float4::operator = ( const _float4& d )
{
	x = d.x;
	y = d.y;
	z = d.z;
	w = d.w;

	return *this;
}

_float4 _float4::operator + ( const _float4& d )
{
	return _float4( x + d.x , y + d.y , z + d.z , w + d.w );
}

//some operators
_float4 _float4::operator - ( const _float4& d )
{
	return _float4( x - d.x , y - d.y , z - d.z , w - d.w );
}

//some operators
_float4 _float4::operator * ( const float factor )
{
	return _float4( x * factor , y * factor , z * factor , w * factor );
}

//some operators
_float4	operator * ( const float factor , _float4& v )
{
	_float4 r;

	r.x = v.x * factor;
	r.y = v.y * factor;
	r.z = v.z * factor;
	r.w = v.w * factor;

	return r;
}

_float4 _float4::operator += ( const _float4& d )
{
	x += d.x;
	y += d.y;
	z += d.z;
	w += d.w;

	return *this;
}

//some operator
_float4 _float4::operator * ( const _float4& d )
{
	_float4 r;
	r.x = x * d.x;
	r.y = y * d.y;
	r.z = z * d.z;
	r.w = w * d.w;

	return r;
}

//some operator
float&  _float4::operator [] ( int index )
{
	switch( index )
	{
	case 0:
		return x;
	case 1:
		return y;
	case 2:
		return z;
	}

	return w;
}

float	length( const _float4& v )
{
	return sqrt( v.x * v.x + v.y * v.y + v.z * v.z );
}

void	normalize( _float4& v )
{
	float len = length( v );

	if( len != 0 )
	{
		v.x /= len;
		v.y /= len;
		v.z /= len;
	}
	v.w = 0;
}

_float4	reflect( _float4& dir , _float4& normal )
{
	dir.w = 0;
	float dotProduct = ( -2.0f ) * dot( dir , normal );

	return dir + dotProduct * normal;
}

//refraction direction
_float4	refract( _float4& dir , _float4& normal , float rate )
{
	_float4 r;

	if( dot( dir , normal ) > 0 )
	{
		normal = -1.0f * normal;
		rate = 1.0f / rate;
	}

	float cos = -1.0f * dot( dir , normal );
	float t = 1 - rate * rate * ( 1 - cos * cos );

	if( t < 0 )
	{
		r = reflect( dir , normal );
	}else
	{
		float cos2 = sqrt( t );
		r = rate * dir + ( rate * cos - cos2 ) * normal ;
	}

	return r;
}

void	saturate( _float4& v )
{
	CLAMP(v.x);
	CLAMP(v.y);
	CLAMP(v.z);
	CLAMP(v.w);
}

//clamp the data
float	clamp( float d )
{
	return min( 1.0f , max( 0.0f , d ) );
}
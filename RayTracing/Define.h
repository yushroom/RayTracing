/*
 *	Programmer:	Jiayin Cao
 *
 *	FileName:	Define.h
 *
 *	Description:	Some usefull macros
 */

#pragma once

//safe release
#define	SAFE_RELEASE(p) {if(p){(p)->Release();(p)=NULL;}}
//safe delete
#define	SAFE_DELETEARRAY(p) {if(p){delete[] (p); (p)=NULL;}}
//safe release cuda memory
#define	SAFE_RELEASE_CUDA(p) {if(p){ cudaFree(p); (p) = 0; }}

//default wheel position
#define	DEFAULT_CURSOR_WHEEL_POS 0

//color
#define RGB_BYTE(r,g,b) ((COLORREF)(((BYTE)(b)|((WORD)((BYTE)(g))<<8))|(((DWORD)(BYTE)(r))<<16)))
#define RGB_FLOAT(r,g,b) RGB_BYTE( ((int)255*r) , ((int)255*g) , ((int)255*b) )
#define RGB_FLOATARRAY(color) RGB_FLOAT( color[0] , color[1] , color[2] )
#define RGB_FLOAT4(v) RGB_FLOAT(v.x,v.y,v.z)
#define CLAMP(value) (value=min(1.0f,max(0.0f,value)))

#define COLOR_R(color) (color>>16)&0x000000ff
#define COLOR_G(color) (color>>8)&0x000000ff
#define	COLOR_B(color) (color)&0x000000ff

//default shader
#define	DEFAULT_SHADER_FOR_ENTITY			0
#define	DEFAULT_SHADER_FOR_ENTITY_FILENAME	L"MeshEffect.fx"
#define	DEFAULT_SHADER_FOR_KDTREE			1
#define	DEFAULT_SHADER_FOR_KDTREE_FILENAME	L"KDTreeEffect.fx"
#define DEFAULT_SHADER_FOR_TRIID			2
#define	DEFAULT_SHADER_FOR_TRIID_FILENAME	L"ShowTriangleIndex.fx"
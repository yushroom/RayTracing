/*
 *	FileName:		Timer.cpp
 *
 *	Programmer:		Jiayin Cao
 *
 *	Description:	A time counter
 */

#include "Timer.h"
#include <windows.h>

//constructor and destructor
Timer::Timer()
{
	//set the default value
	m_Clocks = 0;
	m_Start = 0;

	//get the frequency
	QueryPerformanceFrequency( (LARGE_INTEGER*)&m_Freq );
}

Timer::~Timer()
{
}

//reset the timer
void Timer::Reset()
{
	m_Clocks = 0;
}

//start the counter
void Timer::Start()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&m_Start);
}

//stop the counter
void Timer::Stop()
{
	__int64 stop;
	QueryPerformanceCounter((LARGE_INTEGER*)&stop);

	m_Clocks += stop - m_Start;
	m_Start = 0;
}

//get elapsed time
float Timer::GetElapsedTime()
{
	return (float)1000.0f * ((float)((double)m_Clocks/(double)m_Freq));
}
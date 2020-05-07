using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class Wheel
    {       
        protected string m_ManufacturerName;
        protected float m_CurrentAirPressure;
        protected readonly float r_MaxAirPressure;

        public Wheel(string i_ManufacturerName, float i_CurrentAirPressure, float i_MaxAirPressure)
        {
            m_ManufacturerName = i_ManufacturerName;
            m_CurrentAirPressure = i_CurrentAirPressure;
            r_MaxAirPressure = i_MaxAirPressure;
        }

        internal void InflateWheel(float i_AddedAirPsi)
        {
            float newAirPressure = m_CurrentAirPressure + i_AddedAirPsi;
            if (i_AddedAirPsi < 0 || newAirPressure > r_MaxAirPressure)
            {
                throw new ValueOutOfRangeException(r_MaxAirPressure, 0);
            }
            m_CurrentAirPressure = newAirPressure;
        }

        internal void InflateWheelToMax()
        {
            m_CurrentAirPressure = r_MaxAirPressure;
        }

        public override string ToString()
        {
            string wheelInfo = string.Format(
                                                   "Manufacturer Name: {1}{0}Current Pressure: {2}{0}Max Pressure: {3}{0}",
                                                   Environment.NewLine,
                                                   m_ManufacturerName,
                                                   m_CurrentAirPressure,
                                                   r_MaxAirPressure);
            return wheelInfo;
        }

        public float CurrentAirPressure
        {
            get { return m_CurrentAirPressure; }
            set { m_CurrentAirPressure = value; }
        }

        public float MaxAirPressure
        {
            get { return r_MaxAirPressure; }
        }

        public string ManufacturerName
        {
            get { return m_ManufacturerName; }
        }
    }
}

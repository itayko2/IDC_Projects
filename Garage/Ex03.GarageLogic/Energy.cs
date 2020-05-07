using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public abstract class Energy
    {
        private readonly float r_MaxEnergy;
        private float m_CurrentEnergy;

        public enum eEnergyType
        {
            Soler = 1,
            Octan95 = 2,
            Octan96 = 3,
            Octan98 = 4,
            Electricity = 5
        }

        public Energy(float i_MaxEnergy, float i_CurrentEnergy)
        {
            r_MaxEnergy = i_MaxEnergy;
            m_CurrentEnergy = i_CurrentEnergy;
        }

        public float CurrentEnergy
        {
            get { return m_CurrentEnergy; }
        }

        public float MaxEnergy
        {
            get { return r_MaxEnergy; }
        }
       
        public float MaxRange
        {
            get { return (r_MaxEnergy - m_CurrentEnergy); }
        }

        protected void FillEnergy(float i_AddedEnergy)
        {
            if (i_AddedEnergy <= 0 || m_CurrentEnergy + i_AddedEnergy > r_MaxEnergy)
            {
                throw new ValueOutOfRangeException(r_MaxEnergy - m_CurrentEnergy, 0);
            }
            else if (r_MaxEnergy == m_CurrentEnergy)
            {
                throw new ArgumentOutOfRangeException("Fueltank is full");
            }

            m_CurrentEnergy += i_AddedEnergy;
        }

        public override string ToString()
        {
            string energyInfo = string.Format(
                                               "Current Energy: {1}{0}" + "Max Energy : {2}{0}",
                                               Environment.NewLine,
                                               CurrentEnergy,
                                               MaxEnergy);
            return energyInfo;
        }
    }
}

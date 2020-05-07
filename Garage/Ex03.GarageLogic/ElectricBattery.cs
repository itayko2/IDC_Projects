using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class ElectricBattery : Energy
    {
        public ElectricBattery(float i_BatteryCapacity, float i_RemainingBatteryTime) : base(i_BatteryCapacity, i_RemainingBatteryTime)
        {
        }

        public void Charge(float i_Charge)
        {
            float m_ChargeInHours = i_Charge / 60;
            FillEnergy(m_ChargeInHours);
        }

        public override string ToString()
        {
            string ElectricInfo = string.Format("Energy Type: Electricity{0}", Environment.NewLine);

            return base.ToString() + ElectricInfo;
        }
    }
}

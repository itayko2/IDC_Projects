using System;
using System.Collections.Generic;
using System.Text;

namespace Ex03.GarageLogic
{
    public class FuelTank : Energy
    {
        public enum eFuelType
        {
            Soler = 1,
            Octan95 = 2,
            Octan96 = 3,
            Octan98 = 4
        }

        private eFuelType m_FuelType;

        public FuelTank(float i_FuelTankVolume, float i_RemainingFuel, eFuelType i_FuelType) : base(i_FuelTankVolume, i_RemainingFuel)
        {
            m_FuelType = i_FuelType;
        }

        public void FillTank(float i_FuelToFill, eFuelType i_FuelType)
        {
            if (m_FuelType != i_FuelType)
            {
                throw new ArgumentException("Incompatible Fuel Type");
            }

            FillEnergy(i_FuelToFill);
        }

        public override string ToString()
        {
            string fuelInfo = string.Format("Energy Type: Fuel{0}" + "Fuel type: {1}{0}", Environment.NewLine, m_FuelType);

            return base.ToString() + fuelInfo;
        }

        public eFuelType FuelType
        {
            get { return m_FuelType; }
        }
    }
}

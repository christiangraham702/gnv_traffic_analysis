# Raw Data Schema Information

## Intersection Data (`traffic_data.csv`)

* Shape: (72, 44)
* Columns:

  - `INTERSECT`: object
  - `Period`: object
  - `Time`: object
  - `EB_L`: int64
  - `EB_Th`: int64
  - `EB_R`: int64
  - `WB_L`: int64
  - `WB_Th`: int64
  - `WB_R`: int64
  - `NB_L`: int64
  - `NB_Th`: int64
  - `NB_R`: int64
  - `SB_L`: int64
  - `SB_Th`: int64
  - `SB_R`: int64
  - `Delay_s_veh`: float64
  - `LOS`: object
  - `Max_green_EBL`: float64
  - `Max_green_EBT`: int64
  - `Max_green_WBL`: float64
  - `Max_green_WBT`: int64
  - `Max_green_NBL`: float64
  - `Max_green_NBT`: int64
  - `Max_green_SBL`: float64
  - `Max_green_SBT`: float64
  - `Yellow_duration`: float64
  - `Min_green_EBL`: float64
  - `Min_green_EBT`: int64
  - `Min_green_WBL`: float64
  - `Min_green_WBT`: int64
  - `Min_green_NBL`: float64
  - `Min_green_NBT`: int64
  - `Min_green_SBL`: float64
  - `Min_green_SBT`: float64
  - `Ped_clearence_EW`: int64
  - `Ped_clearence_NS`: int64
  - `Receive_lanes_EB`: int64
  - `Receive_lanes_WB`: int64
  - `Receive_lanes_NB`: float64
  - `Receive_lanes_SB`: float64
  - `Num_lanes_EB`: int64
  - `Num_lanes_WB`: int64
  - `Num_lanes_NB`: int64
  - `Num_lanes_SB`: float64

## Link Data (`lots_traffic_data.csv`)

* Shape: (463, 46)
* Columns:

  - `the_geom`: object
  - `Count_Type`: object
  - `Station`: int64
  - `STREET`: object
  - `BLOCK`: int64
  - `ADT_2016`: int64
  - `ADT_2015`: int64
  - `ADT_2014`: int64
  - `ADT_2013`: int64
  - `ADT_2012`: int64
  - `ADT_2011`: int64
  - `ADT_2010`: int64
  - `ADT_2009`: int64
  - `ADT_2008`: int64
  - `ADT_2007`: int64
  - `ADT_2006`: int64
  - `ADT_2005`: int64
  - `ADT_2004`: int64
  - `ADT_2003`: int64
  - `ADT_2002`: int64
  - `ADT_2001`: int64
  - `ADT_2000`: int64
  - `D1_1314`: object
  - `D2_1314`: object
  - `ADT_1314`: int64
  - `DlyD1_1314`: float64
  - `DlyD2_1314`: float64
  - `Heavy_1314`: float64
  - `PkAM_1314`: int64
  - `AMD1_1314`: float64
  - `AMD2_1314`: float64
  - `PkPM_1314`: int64
  - `PMD1_1314`: float64
  - `PMD2_1314`: float64
  - `D1_1112`: object
  - `D2_1112`: object
  - `ADT_1112`: float64
  - `DlyD1_1112`: float64
  - `DlyD2_1112`: float64
  - `Heavy_1112`: object
  - `PkAM_1112`: int64
  - `AMD1_1112`: float64
  - `AMD2_1112`: float64
  - `PkPM_1112`: int64
  - `PMD1_1112`: float64
  - `PMD2_1112`: float64

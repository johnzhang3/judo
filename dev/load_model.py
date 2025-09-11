import mujoco
from judo import MODEL_PATH

chair_path = "/Users/johnzhang/Documents/research/judo-private/judo/models/xml/spot_components/spot_yellow_chair_ramp.xml"
chair_model = mujoco.MjModel.from_xml_path(chair_path)
chair_data = mujoco.MjData(chair_model)


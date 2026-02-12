import numpy as np
from pulser import Register, Pulse, Sequence
from pulser import ConstantWaveform, InterpolatedWaveform
from pulser.json.abstract_repr.deserializer import deserialize_device
from pasqal_cloud import SDK
from pasqal_cloud.device import DeviceTypeName


# login to Pasqal Cloud account
username = input("Username: ")
password = input("Password: ")
with open("project_id.txt") as file:
    project_id = file.read().strip()

# create Pasqal Cloud remote connection
sdk = SDK(username=username, password=password, project_id=project_id)

# print Fresnel device specs
dev = deserialize_device(sdk.get_device_specs_dict()["FRESNEL"])
print(dev.specs)

# create 5x5 square lattice and define blockade radius
reg = Register.square(side=5, spacing=5, prefix="q").with_automatic_layout(dev)
blockade_radius = 7  # µm
reg.draw(blockade_radius=blockade_radius, draw_half_radius=True)

# generate some random data for testing
num_samples = 5
num_features = 3
data = np.random.rand(num_samples, num_features)

# define pulse sequence with time-dependent global detuning
seq = Sequence(register=reg, device=dev)
time = 6000  # ns
rabi_frequency = dev.rabi_from_blockade(blockade_radius)  # rad/µs
amplitude = ConstantWaveform(duration=time, value=rabi_frequency)
deltas = seq.declare_variable(name="deltas", size=num_features)
detuning = InterpolatedWaveform(duration=time, values=deltas)
pulse = Pulse(amplitude=amplitude, detuning=detuning, phase=0.0)
seq.declare_channel(name="ch", channel_id="rydberg_global")
seq.add(pulse=pulse, channel="ch")

# create jobs and submit to remote device
device_type = DeviceTypeName.EMU_FRESNEL
answer = input(f"\nSubmit a batch of {num_samples} jobs to {device_type}? [y/N] ")
if answer == "y":
    jobs = [{"runs": 500, "variables": {"deltas": list(x)}} for x in data]
    batch = sdk.create_batch(
        jobs=jobs, serialized_sequence=seq.to_abstract_repr(), device_type=device_type
    )
    print("\u2705 Submitted")
else:
    print("\u26d4 Canceled")

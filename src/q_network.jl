

"""
Build a q network given an architecture
"""
function build_q(input::Tensor, arch::QNetworkArchitecture, env::Union{POMDPEnvironment, MDPEnvironment}, scope::String)
    return cnn_to_mlp(input, arch.conv, arch.fc, n_actions(env), scope=scope)
end
using DeepQLearning
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using POMDPs
using POMDPModelTools
using Flux
using Random
using StaticArrays
using Test
import CommonRLInterface

RL = CommonRLInterface

Random.seed!(7)
GLOBAL_RNG = MersenneTwister(1) # for test consistency

function evaluate(mdp::Union{MDP,POMDP}, policy, rng, n_ep=100, max_steps=100)
    avg_r = 0.
    sim = RolloutSimulator(rng=rng, max_steps=max_steps)
    for i=1:n_ep
        DeepQLearning.resetstate!(policy)
        avg_r += simulate(sim, mdp, policy)
    end
    return avg_r/=n_ep
end

function evaluate(env::RL.AbstractEnv, policy, rng, n_ep=100, max_steps=100)
    avg_r = 0.
    for i=1:n_ep
        DeepQLearning.resetstate!(policy)
        r = 0.0
        step = 0
        RL.reset!(env)
        while !RL.terminated(env) && step < max_steps
            a = action(policy, RL.observe(env))
            r += RL.act!(env, a)
            step += 1
        end
        avg_r += r
    end
    return avg_r/=n_ep
end

# @testset "TigerPOMDP DDRQN" begin
    pomdp = TigerPOMDP(0.01, -1.0, 0.1, 0.8, 0.95);
    input_dims = reduce(*, size(convert_o(Vector{Float64}, first(observations(pomdp)), pomdp))) + reduce(*, size(convert_a(Vector{Float64}, first(actions(pomdp)), pomdp)))
    @show input_dims
    model = Chain(x->flattenbatch(x), LSTM(input_dims, 4), Dense(4, length(actions(pomdp))))
    max_steps = 10000
    exploration = EpsGreedyPolicy(pomdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=max_steps/2),
                                  rng=GLOBAL_RNG)
    solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=false, max_steps=max_steps,
                             learning_rate=0.0001, exploration_policy = exploration,
                             log_freq=500, target_update_freq = 1000,
                             recurrence=true,trace_length=10, double_q=true, dueling=true, max_episode_length=100)

    policy = solve(solver, pomdp)
    @test size(actionvalues(policy, true)) == (length(actions(pomdp)),)
# end
% Generates a sparse synaptic connection matrix.
% Returns a synaptic connection lookup table, `synapse_lut`, and the number
% of synaptic connections made, `num_connections`.
%
% A connection is established with a probability defined by `epislon`.
%
% ie. the synapse_id at synapse_lut(pre, post) represents a synaptic connection
% from a pre-synaptic neuron (pre) to a post-synaptic neuron (post).
function [synapse_lut, num_connections] = make_synaptic_connections(num_pre, num_post, epsilon)
    synapse_lut = zeros(num_pre, num_post);  % connection lookup table
    syn_count = 0;  % synapse_id and connection counter
    for pre_neuron = 1:num_pre
        for post_neuron = 1:num_post
            if rand <= epsilon
                syn_count = syn_count + 1;
                synapse_lut(pre_neuron, post_neuron) = syn_count;
            end
        end
    end
    num_connections = syn_count;
end
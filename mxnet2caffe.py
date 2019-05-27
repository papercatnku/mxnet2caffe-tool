# coding: utf-8
import mxnet as mx
import numpy as np
from caffe import layers as L, params as P, to_proto
import caffe as cf
import json
import sys,os

layer_count = 0
inv_short_cf_name_mapping = {}
# some utility functions
def add_layer_to_net_spec(ns, caffe_layer, name, *args, **kwargs):
    kwargs.update({'name':name})
    l = caffe_layer(*args, **kwargs)
    ns.__setattr__(name, l)
    return ns.__getattr__(name)

def add_layer_with_multiple_tops(ns, caffe_layer, lname, ntop, *args, **kwargs):    
    kwargs.update({'name':lname,'ntop':ntop})
    num_in = len(args)-ntop # number of input blobs
    tops = caffe_layer(*args[:num_in], **kwargs)
    for i in xrange(ntop):
        ns.__setattr__(args[num_in+i],tops[i])
    return tops

def remove_tail(src, tail):
    if type(tail) in (list ,tuple):
        for _tail in tail:
            if(_tail == src[-len(_tail):]):
                return src[:-len(_tail)]
        return src
    if(tail == src[-len(tail):]):
        return src[:-len(tail)]
    return src


supported_keys = [
    'Convolution',
    'BatchNorm',
    'FullyConnected',
    'Activation',
    'elemwise_add',
    'elemwise_mul',
    'Reshape',
    'Flatten',
    'transpose',
    'Concat',
    'Deconvolution'
    'LeakyReLU',
    'InstanceNorm', # bn instead for building prototxt
    'Dropout',
]

rn_name_map = {
    'Convolution':'conv',
    'BatchNorm':'bn',
    'FullyConnected':'fc',
    'Activation':'act',
    'elemwise_add':'elt_sum',
    'elemwise_mul':'elt_mul',
    'Reshape':'rshp',
    'Flatten':'flat',
    'transpose':'trsp',
    'Concat':'concat',
    'Deconvolution':'dconv',
    'LeakyReLU':'lReLU',
    'InstanceNorm':'in',
}

def make_rnname(srcname):
    if srcname in rn_name_map:
        return rn_name_map[srcname]
    else:
        return 'uk'

act_mapping = {
    'sigmoid':L.Sigmoid,
    'relu':L.ReLU,
    'tanh':L.Tanh,
}

# convert func
def cvt_prototxt(ns, previous_layers, node, nodes, input_datashps, *args, **kwargs):

    global layer_count
    global inv_short_cf_name_mapping
    # input
    mod_layer_name = remove_tail(node['name'], ['_fwd','-fwd'])
    cf_cur_layer_name = '%s-%.4d'%(make_rnname(node['op']), layer_count)
    inv_short_cf_name_mapping[mod_layer_name] = cf_cur_layer_name

    if (node['op'] == 'null'):
        if node['name'] in input_datashps:
            cvt_layer = add_layer_to_net_spec(
                ns,
                L.Input,
                node['name'],
                input_param={'shape':{'dim':input_datashps[node['name']]}})
            print('add Input layer: %s'%node['name'])
            previous_layers[cf_cur_layer_name] = cvt_layer
        else:
            # print('jump over arg: %s'%node['name'])
            pass
        return ns
    # Convolution
    # TODO: support dialated 
    if(node['op'] == 'Convolution'):
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        convolution_param = {
            'num_output':int(node['attrs']['num_filter']),
            'group':1 if not 'num_group' in node['attrs'] else int(node['attrs']['num_group']),
            'bias_term':not eval(node['attrs']['no_bias'])
        }
        _kernel = eval(node['attrs']['kernel'])
        _pad = [0,0] if not 'pad' in node['attrs'] else eval(node['attrs']['pad'])
#         _stride = eval(node['attrs']['stride'])
        _stride = [1,1] if not 'stride' in node['attrs'] else eval(node['attrs']['stride'])

        convolution_param['kernel_w'] = _kernel[1]
        convolution_param['kernel_h'] = _kernel[0]
        convolution_param['pad_w'] = _pad[1]
        convolution_param['pad_h'] = _pad[0]
        convolution_param['stride_w'] = _stride[1]
        convolution_param['stride_h'] = _stride[0]

        cvt_layer = add_layer_to_net_spec(
            ns,
            L.Convolution,
            cf_cur_layer_name,
            previous_layers[inv_short_cf_name_mapping[input_layer_name]],
            convolution_param=convolution_param)
        previous_layers[cf_cur_layer_name] = cvt_layer
    # Deconvolution
    elif(node['op'] == 'Deconvolution'):
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        deconvolution_param = {
            'num_output':int(node['attrs']['num_filter']),
            'group':1 if not 'num_group' in node['attrs'] else int(node['attrs']['num_group']),
            'bias_term':not eval(node['attrs']['no_bias'])
        }
        _kernel = eval(node['attrs']['kernel'])
        _pad = [0,0] if not 'pad' in node['attrs'] else eval(node['attrs']['pad'])
        _stride = [1,1] if not 'stride' in node['attrs'] else eval(node['attrs']['stride'])
        deconvolution_param['kernel_w'] = _kernel[1]
        deconvolution_param['kernel_h'] = _kernel[0]
        deconvolution_param['pad_w'] = _pad[1]
        deconvolution_param['pad_h'] = _pad[0]
        deconvolution_param['stride_w'] = _stride[1]
        deconvolution_param['stride_h'] = _stride[0]
        cvt_layer = add_layer_to_net_spec(
            ns,
            L.Deconvolution,
            cf_cur_layer_name,
            previous_layers[inv_short_cf_name_mapping[input_layer_name]],
            convolution_param=deconvolution_param)
        previous_layers[cf_cur_layer_name]  = cvt_layer
    # BatchNorm
    # TODO: support non-canonical cases
    elif(node['op'] == 'BatchNorm'):
        # caffe bn
#         batchnorm_param = {
#             'use_global_stats':True,
#             'eps':float(node['attrs']['eps'])
#         }
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        cf_bn_name = cf_cur_layer_name
        cvt_layer_bn = add_layer_to_net_spec(
            ns,
            L.BatchNorm,
            cf_bn_name,
            previous_layers[inv_short_cf_name_mapping[input_layer_name]],
#             batchnorm_param=batchnorm_param
            in_place=False
        )
        # caffe scale
        cf_scale_name = cf_cur_layer_name + '_scale'
        scale_param = {
            'bias_term':True,
            'axis':1
        }
        cvt_layer_scale = add_layer_to_net_spec(
            ns,
            L.Scale,
            cf_scale_name,
            cvt_layer_bn,
            scale_param=scale_param,
            in_place=False
        )
        previous_layers[cf_cur_layer_name]  = cvt_layer_scale
    # Instance Norm, MVN + Scale implementation
    elif(node['op'] == 'InstanceNorm'):
        # caffe MVN
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        cf_in_name = cf_cur_layer_name
        cvt_layer_in =  add_layer_to_net_spec(
            ns,
            L.MVN, # MVN Layer
            cf_in_name,
            previous_layers[inv_short_cf_name_mapping[input_layer_name]],
            mvn_param={'eps':1e-5},
            in_place=False
        )
        # caffe
        cf_scale_name = cf_in_name + '_scale'
        scale_param = {
            'bias_term':True,
            'axis':1
        }
        cvt_layer_scale = add_layer_to_net_spec(
            ns,
            L.Scale,
            cf_scale_name,
            cvt_layer_in,
            scale_param=scale_param,
            in_place=False
        )
        previous_layers[cf_in_name]  = cvt_layer_scale
    # FullyConnected
    # TODO: fix the naive support for fc
    elif(node['op'] == 'FullyConnected'):
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        # consider only flatten case
        cf_input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        inner_product_param = {
            'num_output': int(node['attrs']['num_hidden']),
        }
        if 'no_bias' in node['attrs']:
            inner_product_param['bias_term'] = not eval(node['attrs']['no_bias'])
        cf_innerproduct_layer = add_layer_to_net_spec(
            ns,
            L.InnerProduct,
            cf_cur_layer_name,
            # previous_layers[cf_input_layer_name],
            previous_layers[inv_short_cf_name_mapping[input_layer_name]],
            inner_product_param=inner_product_param)
        previous_layers[cf_cur_layer_name]  = cf_innerproduct_layer
    elif(node['op'] == 'Activation'):
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        if node['attrs']['act_type'] in act_mapping:
            cf_act_layer = add_layer_to_net_spec(
                ns,
                act_mapping[node['attrs']['act_type']],
                cf_cur_layer_name,
                previous_layers[inv_short_cf_name_mapping[input_layer_name]],
                in_place=False
            )
            previous_layers[cf_cur_layer_name]  = cf_act_layer
        else:
            raise NotImplementedError('Activation Type:%s not supportted'%node['attrs']['act_type'])
    elif(node['op'] == 'LeakyReLU'):
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        relu_param = {
            'negative_slope': float(node['attrs']['slope'])
        }
        cvt_leakyrelu_layer = add_layer_to_net_spec(
            ns,
            L.ReLU,
            cf_cur_layer_name,
            previous_layers[inv_short_cf_name_mapping[input_layer_name]],
            relu_param=relu_param,
            in_place=False
        )
        previous_layers[cf_cur_layer_name]  = cvt_leakyrelu_layer

    elif (node['op'] == 'elemwise_add'):
        lhs_name, rhs_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd')), remove_tail(nodes[node['inputs'][1][0]]['name'],'_fwd')
        cf_eltwise_layer = add_layer_to_net_spec(
            ns,
            L.Eltwise,
            cf_cur_layer_name,
            previous_layers[inv_short_cf_name_mapping[lhs_name]],
            previous_layers[inv_short_cf_name_mapping[rhs_name]],
            eltwise_param={'operation':P.Eltwise.SUM}
        )
        previous_layers[cf_cur_layer_name]  = cf_eltwise_layer
    elif (node['op'] == 'elemwise_mul'):
        lhs_name, rhs_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd')), remove_tail(nodes[node['inputs'][1][0]]['name'],'_fwd')
        cf_eltwise_layer = add_layer_to_net_spec(
            ns,
            L.Eltwise,
            cf_cur_layer_name,
            previous_layers[inv_short_cf_name_mapping[lhs_name]],
            previous_layers[inv_short_cf_name_mapping[rhs_name]],
            eltwise_param={'operation':P.Eltwise.PROD}
        )
        previous_layers[cf_cur_layer_name]  = cf_eltwise_layer
    elif (node['op'] == 'Reshape'):
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        reshape_param = {
            'shape':{'dim':list(eval(node['attrs']['shape']))}}        
        cf_rhp_layer = add_layer_to_net_spec(
            ns,
            L.Reshape,
            cf_cur_layer_name,
            previous_layers[inv_short_cf_name_mapping[input_layer_name]],
            reshape_param=reshape_param
        )
        previous_layers[cf_cur_layer_name]  = cf_rhp_layer
    elif (node['op'] == 'Flatten'):
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        cf_flatten_layer = add_layer_to_net_spec(
            ns,
            L.Flatten,
            cf_cur_layer_name,
            previous_layers[inv_short_cf_name_mapping[input_layer_name]]
        )
        previous_layers[cf_cur_layer_name]  = cf_flatten_layer
    # only good 4 NCHW->NHWC ? 
    elif (node['op'] == 'transpose'):
        input_layer_name = remove_tail(nodes[node['inputs'][0][0]]['name'],('_fwd','-fwd'))
        permute_param={'order':list(eval(node['attrs']['axes']))}
        cv_permute_layer = add_layer_to_net_spec(
            ns,
            L.Permute,
            cf_cur_layer_name,
            previous_layers[inv_short_cf_name_mapping[input_layer_name]],
            permute_param=permute_param,
        )
        previous_layers[cf_cur_layer_name]  = cv_permute_layer

    elif(node['op'] == 'Concat'):
        input_layer_names = [
            remove_tail(nodes[x[0]]['name'],'_fwd') for x in node['inputs']
        ]
        in_layers = [ previous_layers[inv_short_cf_name_mapping[x]] for x in input_layer_names]
        concat_param = {'axis':eval(node['attrs']['dim'])}
        _args = {
            'name':mod_layer_name,
            'concat_param':concat_param
        }

        cv_concat_layer = L.Concat(*in_layers, **_args)
        previous_layers[cf_cur_layer_name]  = cv_concat_layer
    else:
        raise NotImplementedError('%s not supportted'%node['op'])

    layer_count +=1
    return ns

def make_caffe_model(ns, src_param_fn, out_prefix):
     
    src_params = mx.nd.load(src_param_fn)
    ns_params= ns.params
    for k, arr in src_params.items():
        if (k[:4] == 'arg:') or (k[:4] == 'aux:'):
            k = k[4:]

        if (k[-7:] == '_weight') or (k[-7:] == '-weight'):
            key_mxnet = k[:-7]
            if not inv_short_cf_name_mapping[key_mxnet] in ns_params:
                print('unused_param: %s'%k)
                continue
            ns.params[inv_short_cf_name_mapping[key_mxnet]][0].data.flat = arr.asnumpy()
            # _bias for Convolution & InnerProduct Layer
        elif (k[-5:] == '_bias') or (k[-5:] == '-bias'):
            key_mxnet = k[:-5]
            if not inv_short_cf_name_mapping[key_mxnet] in ns_params:
                print('unused_param: %s'%k)
                continue
            ns.params[inv_short_cf_name_mapping[key_mxnet]][1].data.flat = arr.asnumpy()
        # _gamma for Scale
        elif (k[-6:] == '_gamma'):
            key_mxnet = k[:-6]
            if not (inv_short_cf_name_mapping[key_mxnet]+ '_scale') in ns_params:
                print('unused_param: %s'%k)
                continue
            ns.params[inv_short_cf_name_mapping[key_mxnet]+ '_scale'][0].data.flat = arr.asnumpy()
        # _beta for scale
        elif (k[-5:] == '_beta'):
            key_mxnet = k[:-5]
            if not (inv_short_cf_name_mapping[key_mxnet]+ '_scale')  in ns_params:
                print('unused_param: %s'%k)
                continue
            ns.params[inv_short_cf_name_mapping[key_mxnet]+ '_scale'][1].data.flat = arr.asnumpy()
        # _moving_mean for bn
        elif (k[-13:] == '_running_mean'):
            key_mxnet = k[:-13]
            if not inv_short_cf_name_mapping[key_mxnet] in ns_params:
                print('unused_param: %s'%k)
                continue
            ns.params[inv_short_cf_name_mapping[key_mxnet]][0].data.flat = arr.asnumpy()
            ns.params[inv_short_cf_name_mapping[key_mxnet]][2].data.flat = 1
        elif (k[-12:] == '_moving_mean'):
            key_mxnet = k[:-12]
            if not inv_short_cf_name_mapping[key_mxnet] in ns_params:
                print('unused_param: %s'%k)
                continue
            ns.params[inv_short_cf_name_mapping[key_mxnet]][0].data.flat = arr.asnumpy()
            ns.params[inv_short_cf_name_mapping[key_mxnet]][2].data.flat = 1
        # _moving_var for bn
        elif (k[-12:] == '_running_var'):
            key_mxnet = k[:-12]
            if not inv_short_cf_name_mapping[key_mxnet] in ns_params:
                print('unused_param: %s'%k)
                continue
            ns.params[inv_short_cf_name_mapping[key_mxnet]][1].data.flat = arr.asnumpy()
        elif (k[-11:] == '_moving_var'):
            key_mxnet = k[:-11]
            if not inv_short_cf_name_mapping[key_mxnet] in ns_params:
                print('unused_param: %s'%k)
                continue
            ns.params[inv_short_cf_name_mapping[key_mxnet]][1].data.flat = arr.asnumpy()
        # unhandled operator:
        else:
            sys.exit('Unknow Mxnet Param: %s'%k)

    ns.save(out_prefix+'.caffemodel')
    return

def saveModelBin(net_prefix):
    srcnet = cf.Net(
        net_prefix + '.prototxt',
        net_prefix + '.caffemodel',
        cf.TEST)
    outfn = net_prefix + '.caffemodel.bin'
    bytes_total =''
    for k,v in srcnet.params.items():
        for p in v:
            bytes_total += p.data.astype(np.float32).tobytes()
    
    with open(outfn, 'wb') as f:
        f.write(bytes_total)


def cvt_end2end(
    src_sym_path,
    src_param_path,
    data_shps,
    out_prefix
):
    global layer_count
    cf.set_mode_cpu()
    layer_count = 0
    inv_short_cf_name_mapping = {}
    with open(src_sym_path, 'rb') as f:
        src_sym_json = json.load(f)

    net = cf.NetSpec()
    nodes = src_sym_json['nodes']
    pl = {}
    
    for i, node in enumerate(nodes):
        net = cvt_prototxt(net, pl, node, nodes, input_datashps=data_shps)

    dir2save = os.path.dirname(out_prefix)
    if not os.path.isdir(dir2save):
        os.makedirs(dir2save)
    
    with open(out_prefix+'.prototxt', 'wb') as f:
        f.write('%s\n'% net.to_proto())
    
    net_out = cf.Net(out_prefix+'.prototxt',cf.TEST)
    make_caffe_model(net_out, src_param_path, out_prefix)
	
    return


if __name__ == '__main__':
    import argparse
    def get_args(arglist=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--sym', type=str, required=True, help='mxnet sym path')
        parser.add_argument('--param', type=str, required=True, help='mxnet param path')
        parser.add_argument('--datashps', type=str, required=True, help='mxnet param path')
        parser.add_argument('--outprefix', type=str, required=True, help='mxnet param path')
        return parser.parse_args() if arglist is None else parser.parse_args(arglist)
    
    def make_datashps(rawstr):
        items_strs = rawstr.split('.')
        dtshps = {}
        for _str in items_strs:
            nm, shp_str = _str.split(':')
            shp = map(int, shp_str.split(','))
            dtshps[nm] = shp
        return dtshps
        
    args = get_args()
    dtshps = make_datashps(args.datashps)
    cvt_end2end(args.sym, args.param, dtshps, args.outprefix)

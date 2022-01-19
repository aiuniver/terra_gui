export default {
    "SGD": [
        {
            "type": "number",
            "name": "optimizer_extra_momentum",
            "label": "Momentum",
            "parse": "optimizer[extra][momentum]",
            "value": 0,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "checkbox",
            "name": "optimizer_extra_nesterov",
            "label": "Nesterov",
            "parse": "optimizer[extra][nesterov]",
            "value": false,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        }
    ],
    "RMSprop": [
        {
            "type": "number",
            "name": "optimizer_extra_rho",
            "label": "RHO",
            "parse": "optimizer[extra][rho]",
            "value": 0,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_momentum",
            "label": "RHO",
            "parse": "optimizer[extra][momentum]",
            "value": 0,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_epsilon",
            "label": "Epsilon",
            "parse": "optimizer[extra][epsilon]",
            "value": 1e-7,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "checkbox",
            "name": "optimizer_extra_centered",
            "label": "Centered",
            "parse": "optimizer[extra][centered]",
            "value": false,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        }
    ],
    "Adam": [
        {
            "type": "number",
            "name": "optimizer_extra_beta_1",
            "label": "Beta 1",
            "parse": "optimizer[extra][beta_1]",
            "value": 0.9,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_beta_2",
            "label": "Beta 2",
            "parse": "optimizer[extra][beta_2]",
            "value": 0.999,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_epsilon",
            "label": "Epsilon",
            "parse": "optimizer[extra][epsilon]",
            "value": 1e-7,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "checkbox",
            "name": "optimizer_extra_amsgrad",
            "label": "Amsgrad",
            "parse": "optimizer[extra][amsgrad]",
            "value": false,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        }
    ],
    "Adadelta": [
        {
            "type": "number",
            "name": "optimizer_extra_rho",
            "label": "RHO",
            "parse": "optimizer[extra][rho]",
            "value": 0.95,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_epsilon",
            "label": "Epsilon",
            "parse": "optimizer[extra][epsilon]",
            "value": 1e-7,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        }
    ],
    "Adagrad": [
        {
            "type": "number",
            "name": "optimizer_extra_initial_accumulator_value",
            "label": "Initial accumulator value",
            "parse": "optimizer[extra][initial_accumulator_value]",
            "value": 0.1,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_epsilon",
            "label": "Epsilon",
            "parse": "optimizer[extra][epsilon]",
            "value": 1e-7,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        }
    ],
    "Adamax": [
        {
            "type": "number",
            "name": "optimizer_extra_beta_1",
            "label": "Beta 1",
            "parse": "optimizer[extra][beta_1]",
            "value": 0.9,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_beta_2",
            "label": "Beta 2",
            "parse": "optimizer[extra][beta_2]",
            "value": 0.999,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_epsilon",
            "label": "Epsilon",
            "parse": "optimizer[extra][epsilon]",
            "value": 1e-7,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        }
    ],
    "Nadam": [
        {
            "type": "number",
            "name": "optimizer_extra_beta_1",
            "label": "Beta 1",
            "parse": "optimizer[extra][beta_1]",
            "value": 0.9,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_beta_2",
            "label": "Beta 2",
            "parse": "optimizer[extra][beta_2]",
            "value": 0.999,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_epsilon",
            "label": "Epsilon",
            "parse": "optimizer[extra][epsilon]",
            "value": 1e-7,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        }
    ],
    "Ftrl": [
        {
            "type": "number",
            "name": "optimizer_extra_learning_rate_power",
            "label": "Learning rate power",
            "parse": "optimizer[extra][learning_rate_power]",
            "value": -0.5,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_initial_accumulator_value",
            "label": "Initial accumulator value",
            "parse": "optimizer[extra][initial_accumulator_value]",
            "value": 0.1,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_l1_regularization_strength",
            "label": "L1 regularization strength",
            "parse": "optimizer[extra][l1_regularization_strength]",
            "value": 0,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_l2_regularization_strength",
            "label": "L2 regularization strength",
            "parse": "optimizer[extra][l2_regularization_strength]",
            "value": 0,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_l2_shrinkage_regularization_strength",
            "label": "L2 shrinkage regularization strength",
            "parse": "optimizer[extra][l2_shrinkage_regularization_strength]",
            "value": 0,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        },
        {
            "type": "number",
            "name": "optimizer_extra_beta",
            "label": "Beta",
            "parse": "optimizer[extra][beta]",
            "value": 0,
            "disabled": false,
            "list": null,
            "fields": null,
            "api": null
        }
    ]
}

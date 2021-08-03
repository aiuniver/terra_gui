<template>
  <div class="block-left">
    <div class="block-left__fab">
      <Fab @click="add" />
    </div>
    <div class="block-left__header">Входные параметры</div>
    <div class="block-left__body">
      <template v-for="({ title, color }, i) of cardLayers">
        <CardLayer
          :title="title + ' ' + (i + 1)"
          :color="color"
          :key="'cardLayersLeft' + i"
          :height="height"
          @click-btn="click($event, i)"
        >
          <Forms :data="main" @change="change" @height="heightForm" />
        </CardLayer>
      </template>
    </div>
  </div>
</template>

<script>
import { getColor } from "../util/color";
import Fab from "../components/forms/Fab.vue";
import Forms from "../components/forms/Forms.vue";
import CardLayer from "../components/card/CardLayer.vue";
export default {
  name: "BlockMainLeft",
  components: {
    Fab,
    CardLayer,
    Forms,
  },
  data: () => ({
    cardLayers: [{ title: "Input", color: "#FFB054" }],
    height: 0,
  }),
  computed: {
    main() {
      const items = [
        {
          type: "checkbox",
          name: "use_bias",
          label: "Use bias",
          parse: "parameters[extra][use_bias]",
          value: true,
          list: null,
        },
        {
          type: "checkbox",
          name: "use_bias",
          label: "Use bias",
          parse: "parameters[extra][use_bias]",
          value: true,
          list: null,
        },
        {
          type: "checkbox",
          name: "use_bias",
          label: "Use bias",
          parse: "parameters[extra][use_bias]",
          value: true,
          list: null,
        },
        {
          type: "checkbox",
          name: "use_bias",
          label: "Use bias",
          parse: "parameters[extra][use_bias]",
          value: true,
          list: null,
        },

        {
          type: "select",
          name: "bias_initializer",
          label: "Bias initializer",
          parse: "parameters[extra][bias_initializer]",
          value: "zeros",
          list: [
            {
              value: "random_normal",
              label: "Random normal",
            },
            {
              value: "random_uniform",
              label: "Random uniform",
            },
            {
              value: "truncated_normal",
              label: "Truncated normal",
            },
            {
              value: "zeros",
              label: "Zeros",
            },
            {
              value: "ones",
              label: "Ones",
            },
            {
              value: "glorot_normal",
              label: "Glorot normal",
            },
            {
              value: "glorot_uniform",
              label: "Glorot uniform",
            },
            {
              value: "uniform",
              label: "Uniform",
            },
            {
              value: "identity",
              label: "Identity",
            },
            {
              value: "orthogonal",
              label: "Orthogonal",
            },
            {
              value: "constant",
              label: "Constant",
            },
            {
              value: "variance_scaling",
              label: "Variance scaling",
            },
            {
              value: "lecun_normal",
              label: "Lecun normal",
            },
            {
              value: "lecun_uniform",
              label: "Lecun uniform",
            },
            {
              value: "he_normal",
              label: "He normal",
            },
            {
              value: "he_uniform",
              label: "He uniform",
            },
          ],
        },
        {
          type: "select",
          name: "kernel_regularizer",
          label: "Kernel regularizer",
          parse: "parameters[extra][kernel_regularizer]",
          value: "",
          list: [
            {
              value: "l1",
              label: "L1",
            },
            {
              value: "l2",
              label: "L2",
            },
            {
              value: "l1_l2",
              label: "L1 l2",
            },
          ],
        },
        {
          type: "select",
          name: "bias_regularizer",
          label: "Bias regularizer",
          parse: "parameters[extra][bias_regularizer]",
          value: "",
          list: [
            {
              value: "l1",
              label: "L1",
            },
            {
              value: "l2",
              label: "L2",
            },
            {
              value: "l1_l2",
              label: "L1 l2",
            },
          ],
        },
        {
          type: "select",
          name: "activity_regularizer",
          label: "Activity regularizer",
          parse: "parameters[extra][activity_regularizer]",
          value: "",
          list: [
            {
              value: "l1",
              label: "L1",
            },
            {
              value: "l2",
              label: "L2",
            },
            {
              value: "l1_l2",
              label: "L1 l2",
            },
          ],
        },
        {
          type: "select",
          name: "kernel_constraint",
          label: "Kernel constraint",
          parse: "parameters[extra][kernel_constraint]",
          value: "",
          list: [
            {
              value: "max_norm",
              label: "Max norm",
            },
            {
              value: "min_max_norm",
              label: "Min max norm",
            },
            {
              value: "non_neg",
              label: "Non neg",
            },
            {
              value: "unit_norm",
              label: "Unit norm",
            },
            {
              value: "radial_constraint",
              label: "Radial constraint",
            },
          ],
        },
        {
          type: "select",
          name: "bias_constraint",
          label: "Bias constraint",
          parse: "parameters[extra][bias_constraint]",
          value: "",
          list: [
            {
              value: "max_norm",
              label: "Max norm",
            },
            {
              value: "min_max_norm",
              label: "Min max norm",
            },
            {
              value: "non_neg",
              label: "Non neg",
            },
            {
              value: "unit_norm",
              label: "Unit norm",
            },
            {
              value: "radial_constraint",
              label: "Radial constraint",
            },
          ],
        },
      ];
      const value = {};
      const type = "main";
      return { type, items, value };
    },
  },
  methods: {
    add() {
      this.cardLayers.push({ title: "Input", color: getColor() });
    },
    click(comm, index) {
      console.log(comm, index);
      if (comm === "remove") {
        this.cardLayers = this.cardLayers.filter((_, i) => {
          return i !== index;
        });
      }
    },
    change(e) {
      console.log(e);
    },
    heightForm(value) {
      // console.log(value, this.$el.clientHeight);
      // const clearHeight = this.$el.clientHeight - 56;
      console.log(value,  this.$el.clientHeight);
      // this.height = value > clearHeight ? clearHeight : value + 56;
      // this.height = clearHeight
    },
  },
};
</script>

<style lang="scss" scoped>
.block-left {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  height: 100%;
  &__header {
    position: absolute;
    height: 24px;
    width: 100%;
    top: 0;
    background: #242f3d;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
    display: flex;
    align-items: center;
    text-align: center;
    color: #ffffff;
    padding: 4px 16px;
    justify-content: flex-end;
  }
  &__body {
    display: flex;
    padding: 40px 70px 16px 16px;
    width: 100%;
    position: relative;
    justify-content: flex-end;
    overflow: auto;
    // height: 100%;
  }
  &__fab {
    position: absolute;
    right: 16px;
    top: 40px;
    z-index: 100;
  }
}
</style>
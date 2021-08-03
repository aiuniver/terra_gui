<template>
  <div class="block-right">
    <div class="block-right__fab">
      <Fab @click="add" />
    </div>
    <div class="block-right__header">Выходные параметры</div>
    <div class="block-right__body">
      <template v-for="({ title, color }, i) of cardLayers">
        <CardLayer
          :title="title + ' ' + (i + 1)"
          :color="color"
          :key="'cardLayersRight' + i"
          @click-btn="click($event, i)"
        >
          <Forms :data="main" @change="change" />
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
  name: "BlockMainRight",
  components: {
    Fab,
    CardLayer,
    Forms
  },
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
  data: () => ({
    cardLayers: [{ title: "Name", color: "#8e51f2" }],
  }),
  methods: {
    add() {
      this.cardLayers.unshift({ title: "Name", color: getColor() });
    },
    change(e) {
      console.log(e)
    }
  },
};
</script>

<style lang="scss" scoped>
.block-right {
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
  }
  &__body {
    display: flex;
    padding: 40px 16px 16px 72px;
    width: 100%;
    // height: 100%;
    position: relative;
    justify-content: flex-start;
    overflow: auto
  }
  &__fab {
    position: absolute;
    left: 16px;
    top: 40px;
    z-index: 100;
  }
}
</style>
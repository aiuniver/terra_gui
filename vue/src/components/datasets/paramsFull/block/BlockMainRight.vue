<template>
  <div class="block-right">
    <div class="block-right__fab">
      <Fab @click="add" />
    </div>
    <div class="block-right__header">Выходные параметры</div>
    <div class="block-right__body">
      <scrollbar :ops="ops" ref="scrollRight">
        <div class="block-right__body--inner" :style="height">
          <div class="block-right__body--empty"></div>
          <template v-for="({ title, color }, i) of cardLayers">
            <CardLayer
              :title="title + ' ' + (i + 1)"
              :color="color"
              :key="'cardLayersRight' + i"
              @click-btn="click($event, i)"
            >
              <!-- <t-forms :data="main" @change="change" /> -->
            </CardLayer>
          </template>
        </div>
      </scrollbar>
    </div>
  </div>
</template>

<script>
import { getColor } from "../util/color";
import Fab from "../components/forms/Fab.vue";
import CardLayer from "../components/card/CardLayer.vue";
export default {
  name: "BlockMainRight",
  components: {
    Fab,
    CardLayer,
  },
  data: () => ({
    cardLayers: [{ title: "Name", color: "#8e51f2" }],
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
      rail: {
        gutterOfEnds: "6px",
      },
    },
  }),
  computed: {
    height() {
      const height = this.$store.getters["settings/height"]({
        clean: true,
        padding: 172 + 90 + 62,
      });
      console.log(height);
      return height;
    },
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
      this.cardLayers.unshift({ title: "Input", color: getColor() });
      // this.$nextTick(() => {
      //   this.$refs.scrollRight.scrollTo(
      //     {
      //       x: "100%",
      //     },
      //     100
      //   );
      // });
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
    width: 100%;
    /* position: absolute; */
    /* height: 250px; */
    /* top: 34px; */
    padding: 40px 0px 0px 0px;
    /* right: 70px; */
    overflow: auto;
    &--inner {
      display: flex;
      width: 100%;
      justify-content: flex-start;
      overflow: auto;
      // position: absolute;
      height: 100%;
    }
    &--empty {
      height: 100%;
      width: 70px;
    }
  }
  &__fab {
    position: absolute;
    left: 16px;
    top: 40px;
    z-index: 100;
  }
}
</style>
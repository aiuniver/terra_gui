<template>
  <div class="block-right">
    <div class="block-right__fab">
      <Fab @click="add" />
    </div>
    <div class="block-right__header">Выходные параметры</div>
    <div class="block-right__body">
      <template v-for="({ title, color }, i) of cardLayers">
        <CardLayer :title="title" :color="color" :key="'cardLayersRight' + i" />
      </template>
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
  }),
  methods: {
    add() {
      this.cardLayers.unshift({ title: "Name", color: getColor() });
    },
    getColor() {
      var letters = "0123456789ABCDEF";
      var color = "#";
      for (var i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
      }
      return color;
    },
  },
};
</script>

<style lang="scss" scoped>
.block-right {
  flex-basis: 50.000004%;
  max-width: 50.000004%;
  flex: 0 0 auto;
  position: relative;
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
    height: 100%;
    position: relative;
    justify-content: flex-start;
  }
  &__fab {
    position: absolute;
    left: 16px;
    top: 40px;
    z-index: 100;
  }
}
</style>
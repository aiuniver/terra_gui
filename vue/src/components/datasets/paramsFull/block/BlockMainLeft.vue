<template>
  <div class="block-left">
    <div class="block-left__fab">
      <Fab @click="add"/>
    </div>
    <div class="block-left__header">Входные параметры</div>
    <div class="block-left__body">
      <template v-for="({ title, color}, i) of cardLayers">
        <CardLayer :title="title" :color="color" :key="'cardLayersLeft' + i" />
      </template>
    </div>
  </div>
</template>

<script>
import Fab from "../components/forms/Fab.vue";
import CardLayer from "../components/card/CardLayer.vue";
export default {
  name: "BlockMainLeft",
  components: {
    Fab,
    CardLayer,
  },
  data: () => ({
    cardLayers: [
      { title: "Name", color: "#FFB054" },
      { title: "Name", color: "#89d764" },
    ],
  }),
  methods: {
    add() {
      this.cardLayers.push({ title: "Name", color: this.getColor() })
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
.block-left {
  display: flex;
  flex-direction: column;
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
    justify-content: flex-end;
  }
  &__body {
    display: flex;
    padding: 40px 70px 16px 16px;
    width: 100%;
    position: relative;
    justify-content: flex-end;
    overflow: auto;
  }
  &__fab {
    position: absolute;
    right: 16px;
    top: 40px;
    z-index: 100;
  }
}
</style>
<template>
  <div class="card-layer">
    <div class="card-layer__header" :style="bg">Названия слоя</div>
    <div class="card-layer__body"></div>
  </div>
</template>

<script>
export default {
  name: "card-layer",
  props: {
    color: {
			type: String,
			default: '#242f3d'
		},
    title: String,
  },
  computed: {
		bg() {
			return { backgroundColor: this.color }
		},
	},
};
</script>

<style lang="scss" scoped>
.card-layer {
	flex: 0 0 240px;
  position: relative;
  height: 150px;
  border-radius: 4px;
  border: 1px solid #6C7883;
  background-color: #242F3D;
  margin: 0 5px;
  min-height: 150px;

  &__header {
    position: absolute;
    top: 0;
    height: 24px;
    background-color: #6C7883;
    width: 100%;
    border-radius: 3px 3px 0 0;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
    display: flex;
    align-items: center;
    padding: 2px 6px 4px 6px;
  }
}
</style>
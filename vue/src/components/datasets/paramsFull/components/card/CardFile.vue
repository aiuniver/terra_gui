<template>
  <div class="card-file" :style="bc">
    <div class="card-file__header"></div>
    <div class="card-file__body"></div>
    <div class="card-file__footer" :style="bg">{{ title }}</div>
  </div>
</template>

<script>
export default {
  name: "card-file",
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
		bc() {
			return { borderColor: this.color }
		}
	},
};
</script>

<style lang="scss" scoped>
.card-file {
	flex: 0 0 150px;
  position: relative;
  width: 150px;
  height: 150px;
  border-radius: 4px;
  border: #242f3d solid 1px;
  background-color: #17212b;
  margin: 0 5px;
  &__footer {
    position: absolute;
    bottom: 0;
    height: 24px;
    background-color: #242f3d;
    width: 100%;
    border-radius: 0 0 3px 3px;
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
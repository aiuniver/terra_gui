<template>
  <div class="t-large-image">
    <div class="t-large-image__mask" @click="close"></div>
    <div class="t-large-image__fixed">
      <i class="ci-icon ci-close_big" @click="close"/>
      <img width="auto" :height="600" :src="imgSrc" :alt="'value'" :key="imgSrc" />
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'

export default {
	name: 't-large-image',
	computed: {
		...mapGetters({
			imgSrc: 'trainings/getLargeImg'
		})
	},
	methods: {
		close() {
			this.$store.dispatch('trainings/setLargeImg')
		}
	}
}
</script>

<style lang="scss" scoped>
.t-large-image {
  height: 300px;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0 20px;
  &__fixed {
    z-index: 701;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    padding: 20px;
    background-color: #17212b;
    border-radius: 6px;
    border: 1px solid #6c7883;
    cursor: default;
    img {
      border-radius: 6px;
    }
    i {
      position: absolute;
      top: -20px;
      right: -20px;
      font-size: 30px;
      background-color: #17212b;
      border-radius: 50%;
      border: 1px solid #6c7883;
      padding: 5px;
      color: #6c7883;
      cursor: pointer;
      &:hover {
        color: #b5bbc0;
      }
    }
  }
  &__mask {
    position: fixed;
    z-index: 700;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    background-color: #0e1621e6;
    cursor: default;
  }
}
</style>
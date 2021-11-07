<template>
  <div class="img-card">
    <img class="img-card__image" :src="src">
  </div>
</template>

<script>
import {mapGetters} from "vuex";

export default {
  name: "ImgCard",
  props: {
    imgUrl: {
      type: String,
      default: 'img.png'
    },
  },
  computed: {
     ...mapGetters({
      id: 'deploy/getRandId',
    }),
    src() {
      return `/_media/blank/?path=${this.imgUrl}&r=${this.id}`
    }
  }
}
</script>

<style scoped>
.img-card__image{
  border: 1px solid #6C7883;
  box-sizing: border-box;
  border-radius: 4px;
  width: 224px;
}
</style>
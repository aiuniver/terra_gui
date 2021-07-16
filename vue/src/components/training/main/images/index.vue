<template>
  <div class="images">
    <div class="images__title">Изображения</div>
    <div class="images__content">
      <div class="images__content--title">{{ title }}</div>
      <div class="images__content--inner">
        <CardImage v-for="({image, info}, i) of values" :key="i" :image="image" :info="info" />
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import CardImage from "./CardImage.vue";
export default {
  components: {
    CardImage,
  },
  computed: {
    ...mapGetters({
      images: 'trainings/getImages' 
    }),
    title() {
      return this.images.title || ''
    },
    values() {
      return this.images.values || []
    }
  }
};
</script>

<style lang="scss" scoped>
.images {
  &__content {
    margin-top: 10px;
    &--title {
      line-height: 1;
      margin: 0 0 10px 0;
      padding: 10px 20px;
      color: #a7bed3;
      font-size: 0.875rem;
      font-weight: 600;
      user-select: none;
      background: #0e1621;
      border-radius: 4px;
    }
    &--inner {
      margin: 0 -5px -10px -5px;
      display: -webkit-box;
      display: -moz-box;
      display: -ms-flexbox;
      display: -webkit-flex;
      display: flex;
      -webkit-box-direction: normal;
      -moz-box-direction: normal;
      -webkit-box-orient: horizontal;
      -moz-box-orient: horizontal;
      -webkit-flex-direction: row;
      -ms-flex-direction: row;
      flex-direction: row;
      -webkit-flex-wrap: wrap;
      -ms-flex-wrap: wrap;
      flex-wrap: wrap;
      -webkit-box-pack: start;
      -moz-box-pack: start;
      -webkit-justify-content: flex-start;
      -ms-flex-pack: start;
      justify-content: flex-start;
      -webkit-align-content: stretch;
      -ms-flex-line-pack: stretch;
      align-content: stretch;
      -webkit-box-align: stretch;
      -moz-box-align: stretch;
      -webkit-align-items: stretch;
      -ms-flex-align: stretch;
      align-items: stretch;
    }
  }
}
</style>
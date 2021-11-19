<template>
  <div v-if="info" class="info">
    <div class="info__content" v-html="info"></div>
  </div>
</template>

<script>
export default {
  name: 'info-block',
  props: {
    info: String,
  },
};
</script>

<style lang="scss">
.info {
  font-size: 12px;
  line-height: 140%;
  color: #a7bed3;
  &__title {
    background-color: #0e1621;
    margin: 10px -20px;
    padding: 4px 20px;
    font-size: 0.75rem;
    color: #ffffff;
  }
  &__content {
    b {
      color: #ffffff;
    }
    h3 {
      font-weight: normal;
      line-height: 140%;
      color: #ffffff;
    }
    p {
      font-size: 12px;
      color: inherit;
    }
    ol {
      margin: 10px 0;
      list-style-type: style;
      list-style-position: inside;
      li {
        color: inherit;
        margin-bottom: 10px;
      }
    }
  }
}
</style>
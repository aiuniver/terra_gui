<template>
<div class="card">
  <div class="card__content">
    <div class="card__original">
      <ImgCard v-if="original.type == 'image'"/>
      <TextCard v-if="original.type == 'text'" :text="original.data" :style="originaltextStyle"/>
    </div>
    <div class="card__result">
      <ImgCard v-if="result.type == 'image'"/>
      <TextCard v-if="result.type == 'text'" :text="result.data"/>
    </div>
  </div>
  <div class="card__reload"><button class="btn-reload"><i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i></button></div>
</div>
</template>

<script>
import ImgCard from "./cards/ImgCard";
import TextCard from "./cards/TextCard";
export default {
  name: "IndexCard",
  components: {
    ImgCard,
    TextCard,
  },
  data: () => ({
    originaltextStyle: {
      width: "600px",
      color: "#A7BED3",
      padding: "10px 25px 12px 12px"
    }
  }),
  props: {
    original: {
      type: Object,
      default: () => ({})
    },
    result: {
      type: Object,
      default: () => ({})
    }
  },
}
</script>

<style lang="scss" scoped>
.card__reload{
  padding-left: 5px;
}
.card{
  padding: 15px 15px 15px 0;
  display: flex;
}
.card__original{
  background: #242F3D;
}
.card__result{
  padding-top: 6px;
}
.btn-reload{
    width: 32px;
    height: 32px;
    i{
      position: absolute;
      margin-left: -8px;
      margin-top: -13px;
      width: 16px;
    }
  }
</style>
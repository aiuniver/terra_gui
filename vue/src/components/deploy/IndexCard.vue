<template>
<div class="card">
  <div class="content">
    <div class="original">
      <ImgCard v-if="DataBlock.original.type == 'image'"/>
    </div>
    <div class="result">
      <ImgCard v-if="DataBlock.result.type == 'image'"/>
      <TextCard v-if="DataBlock.result.type == 'text'" :text="DataBlock.result.data"/>
    </div>
  </div>
  <div class="reload"><button><i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i></button></div>
</div>
</template>

<script>
import ImgCard from "./cards/ImgCard";
import TextCard from "./cards/TextCard";
export default {
  name: "IndexCard",
  components: {
    ImgCard,
    TextCard
  },
  props: {
    DataBlock: {
      type: Object,
      default: () => ({})
    }
  },
}
</script>

<style lang="scss" scoped>
.reload{
  padding-left: 5px;
  button{
    width: 32px;
    height: 32px;
  }
  i{
    position: absolute;
    margin-left: -8px;
    margin-top: -13px;
    width: 16px;
  }
}
.card{
  padding: 15px 15px 15px 0;
  display: flex;
}
.original{
  background: #242F3D;
}
.result{
  padding-top: 6px;
}
</style>
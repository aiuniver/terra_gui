<template>
  <div class="t-predict-text">
    <p class="t-predict-text__text">{{ text }}</p>
    <span v-if="length" class="t-predict-text__more" @click="show(idx)">{{ textBtn[Number(isExpanded)] }}</span>
    <!-- <button v-if="length" @click="show">{{ textBtn[Number(isShow)] }}</button> -->
  </div>
</template>

<script>
export default {
  name: 'table-text',
  props: {
    value: {
      type: String,
      default: '',
    },
    color_mark: {
      type: Array,
      default: () => [],
    },
    idx: {
      default: null
    }
  },
  data: () => ({
    // text: '',
    isShow: false,
    textBtn: ["Показать больше", "Скрыть"]
  }),
  mounted(){
    // this.text = this.length ? this.value.substring(0, 49) + "..." : this.value
  },
  computed:{
    length(){
      return this.value.length >= 50 
    },
    text() {
      if (this.isExpanded) return this.value
      return this.value.substring(0, 49) + "..."
    },
    isExpanded() {
      return this.$store.getters['trainings/getExpandedIdx'].includes(this.idx)
    }
  },
  methods:{
    show(idx){
      this.isShow = !this.isShow
      this.$store.dispatch('trainings/setExpandedIdx', { value: this.isShow, idx })
    }
  },
  watch: {
    isExpanded(newVal) {
      this.isShow = newVal
    }
  }
};
</script>

<style lang="scss" scoped>
.t-predict-text {
  display: flex;
  height: 100%;
  justify-content: center;
  align-items: center;
  width: 400px;
  padding: 5px;
  flex-direction: column;
  &__text {
    text-align: center;
    margin-bottom: 10px;
  }
  &__more {
    user-select: none;
    cursor: pointer;
    color: #65b9f4;
    font-size: 14px;
  }
}
</style>
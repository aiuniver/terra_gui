<template>
  <div class="t-predict-text">
    <p class="t-predict-text__text">{{ text }}</p>
    <button v-if="length" @click.native="show">Показать больше</button>
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
  },
  data: () => ({
    text: '',
    isShow: false,
  }),
  mounted(){
    this.text = this.length ? this.value.substring(0, 99) + "..." : this.value
  },
  computed:{
    length(){
      return this.value.length >= 100 
    }
  },
  methods:{
    show(){
      if(this.length && this.isShow)
        this.text = this.value
      else
        this.text = this.value.substring(0, 99) + "..."

      this.isShow = !this.isShow
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
  &__text {
    text-align: center;
  }
}
</style>
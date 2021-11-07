<template>
  <div class="t-predict-text">
    <p :class="['t-predict-text__text', color]" :style="{marginTop: length ? '10px' : '' }">{{ text }}</p>
    <span v-if="length" class="t-predict-text__more" @click="show">{{ textBtn[Number(isShow)] }}</span>
  </div>
</template>

<script>
export default {
  name: 'TableStr',
  props: {
    value: {
      type: [String, Number],
      default: '',
    },
    color_mark: {
      type: String,
      default: '',
    },
    tagsColor: {
      type: String,
      default: '',
    },
  },
  data: () => ({
    text: '',
    isShow: false,
    textBtn: ["Показать больше", "Скрыть"]
  }),
  mounted(){
    this.text = this.length ? this.value.substring(0, 49) + "..." : this.value
  },
  computed: {
    color() {
      return `t-predict-text__text--${this.color_mark}`
    },
    length(){
      return this.value.length >= 50 
    }
  },
  methods:{
    show(){
      this.isShow = !this.isShow
      this.text = this.isShow? this.value : this.value.substring(0, 49) + "..."
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
  flex-direction: column;
  padding: 5px;

  &__text {
    text-align: center;
    &--success {
      color: green;
    }
    &--wrong {
      color: orange;
    }
  }
  &__more {
    user-select: none;
    cursor: pointer;
    color: #65b9f4;
    font-size: 14px;
  }
}

</style>

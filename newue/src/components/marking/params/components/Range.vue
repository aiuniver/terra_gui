<template>
    <div class="t-range">
        <div class="t-range__content">
            <div class="t-range__left"><p>{{ minMax[0] }}</p></div>
            <div class="t-range__right"><p>{{ minMax[1] }}</p></div>
        </div>
        <div class="t-range__line">
            <div v-for="(item, idx) in [...value]"
            :key="idx"
            class="t-range__line-to"
            :style="`background: ${colors[idx]}; width: ${item}%`"><p v-if="showValue">{{ minMax[1] * item / 100 }}</p></div>
        </div>
    </div>
</template>

<script>
export default {
    name: 't-range',
    props: {
        minMax: {
            type: Array,
            default: () => [0, 900]
        },
        showValue: {
            type: Boolean,
            default: false
        },
        value: {
            type: [Array, Number],
            default: 50
        }
    },
    data: () => ({
        colors: [
            '#FFB054', '#89D764', '#71CE45', '#54A3FF',
            '#FF9D2A', '#71CE45', '#2A8CFF', '#762BEF',
            '#FE8900', '#5DBB31', '#0075FE', '#6011E2',
            '#D47200', '#4E9C29', '#0062D4', '#500EBC',
            '#A95C00', '#3E7D21', '#004EAA', '#400B96',
            '#7F4500', '#2F5E18', '#003B7F', '#300871'
        ]
    })
}
</script>

<style lang="scss" scoped>
.t-range {
    width: 100%;
    position: relative;
    p {
        color: #a7bed3;
        font-family: 'Open Sans';
        font-style: normal;
        font-weight: normal;
        font-size: 12px;
        line-height: 24px;
    }
    &__line {
      background: #a7bed3;
      border-radius: 3px;
      height: 6px;
      display: flex;

      &-to {
        height: 6px;
        border-radius: 3px;
        position: relative;
        p {
            position: absolute;
            bottom: 100%;
            right: 0;
            transform: translateX(50%);
        }
      }
    }

    &__content {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2px;
    }
  }
</style>
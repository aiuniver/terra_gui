<template>
  <div class="card">
    <div class="cord__body icon-model-load">
      <div class="card__body--image">
        <img  width="100%" src="/imgs/bmw.jpg" alt="images">
      </div>
    </div>
    <div class="card__footer">
      <span> Sloy</span>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  name: "CardFile",
  props: {
    name: {
      type: String,
      required: true,
    },
  },
  data: () => ({
    select: "",
  }),
  computed: {
    ...mapGetters({
      settings: "datasets/getSettings",
    }),
  },
};
</script>

<style lang="scss" scoped>
@import "@/at-ui/scss/variables/index.scss";

.card {
  position: relative;
  max-width: 200px;
  height: 200px;
  border: solid 2px #6c7883;
  border-radius: 8px;
  background-color: #17212b;
  transition: all 0.3s;
  transition: all 0.3s;
  margin: 10px;

  &:not(.card--no-hover):hover {
    border-color: $card-border-color-hover;
    box-shadow: 1px 0 16px 0 $shadow-color;
  }

  &__body--image {
    margin-top: 20px;
  }

  &__footer {
    position: absolute;
    bottom: 0px;
    height: 30px;
    background-color: #242f3d;
    width: 100%;
    span {
      padding: 5px 10px;;

    }
  }

  /* element */
  &__head {
    padding: 0 24px;
    height: $card-head-height;
    line-height: $card-head-height;
    border-bottom: 1px solid $card-border-color;
  }
  &__title {
    display: inline-block;
  }
  &__extra {
    float: right;
  }
  &__body {
    padding: 24px;

    &--loading {
      span {
        display: inline-block;
        margin: 5px 1%;
        height: 14px;
        border-radius: 2px;
        background: linear-gradient(
          90deg,
          rgba(192, 198, 206, 0.12),
          rgba(192, 198, 206, 0.2),
          rgba(192, 198, 206, 0.12)
        );
        background-size: 600% 600%;
        animation: card-loading 1.4s ease infinite;
      }
    }
  }

  /* modifier */
  &--bordered {
    border: 1px solid $card-border-color;
  }
}

@keyframes card-loading {
  0%,
  to {
    background-position: 0 50%;
  }
  50% {
    background-position: 100% 50%;
  }
}
</style>

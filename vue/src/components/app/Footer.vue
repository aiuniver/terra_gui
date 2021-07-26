<template>
  <div class="footer">
    <div class="footer__message">
      <div :class="['footer__message--text', color]">
          {{ message }}
      </div>
    </div>
    <div class="footer__progress">
      <div class="footer__progress--item">
        <i :style="{ width: progress + '%' }"><span></span></i><span></span>
      </div>
    </div>
    <div class="footer__state">
      <div class="footer__state--icon" :style="style"></div>
      <div class="footer__state--text">{{ protsessor.type }}</div>
    </div>
    <div class="footer__copyright">
      {{
        `Copyright © «Университет искусственного интеллекта», ${new Date().getFullYear()}`
      }}
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  computed: {
    ...mapGetters({
      message: "messages/getMessage",
      color: "messages/getColor",
      progress: "messages/getProgress",
      project: "projects/getProject",
    }),
    protsessor() {
      return this.project?.hardware || "";
    },
    style() {
      return { backgroundColor: "#" + this.protsessor.color };
    },
  },
};
</script>

<style lang="scss" scoped>
.error {
  color: #ffb054 !important;
}
.success {
  color: #0f0 !important;
}
.footer {
  border-top: solid 1px #0e1621;
}
.state {
  text-transform: uppercase;
}
.footer {
  color: #a7bed3;
  background: #17212b;
  width: 100%;
  line-height: 30px;
  position: fixed;
  left: 0;
  bottom: 30px;
  z-index: 900;
  font-size: 0.75rem;
  white-space: nowrap;
  user-select: none;
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
  -webkit-flex-wrap: nowrap;
  -ms-flex-wrap: nowrap;
  flex-wrap: nowrap;
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
  &__message {
    border-right: #0e1621 1px solid;
    overflow: hidden;
    -webkit-box-ordinal-group: 1;
    -moz-box-ordinal-group: 1;
    -webkit-order: 0;
    -ms-flex-order: 0;
    order: 0;
    -webkit-box-flex: 1;
    -moz-box-flex: 1;
    -webkit-flex: 1 1 auto;
    -ms-flex: 1 1 auto;
    flex: 1 1 auto;
    -webkit-align-self: auto;
    -ms-flex-item-align: auto;
    align-self: auto;
    &--text {
      overflow: hidden;
      text-overflow: ellipsis;
      padding: 0 20px;
    }
  }
  &__state {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 59px;
    font-weight: 600;
    &--text {
      text-transform: uppercase;
    }
    &--icon {
      display: inline-block;
      content: "";
      width: 10px;
      height: 10px;
      margin: 0 5px 0 0;
      border-radius: 50%;
      vertical-align: middle;
    }
  }
  &__progress {
    border-right: #0e1621 1px solid;
    &--item {
      width: 339px;
      height: 100%;
      padding: 3px;
      position: relative;
      > i {
        background-color: #242f3d;
        display: block;
        width: 0;
        height: 100%;
        position: relative;
        z-index: 1;
        font-style: normal;
        overflow: hidden;
      }
    }
  }
  &__copyright {
    position: fixed;
    width: 100%;
    left: 0;
    bottom: 0;
    z-index: 901;
    user-select: none;
    line-height: 30px;
    padding: 0 10px;
    color: #a7bed3;
    font-size: 0.6875rem;
    text-align: right;
    background-color: #0e1621;
  }
}
</style>